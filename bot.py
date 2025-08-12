from __future__ import annotations

import os
import io
import json
import random
import asyncio
import datetime
from typing import Dict, Optional, Literal, Tuple
from zoneinfo import ZoneInfo

import discord
from discord import app_commands
from discord.ext import tasks

import aiohttp
import re

# =================================
#  環境変数を読み込む設定
# =================================
TOKEN = os.getenv("DISCORD_TOKEN")
GUILD_ID = os.getenv("GUILD_ID")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

DEFAULT_TZ = os.getenv("BOT_TZ", "Asia/Tokyo")
TZ = ZoneInfo(DEFAULT_TZ)

# =================================
#  Botの基本的な設定
# =================================
intents = discord.Intents.default()
intents.message_content = True
intents.members = True

DATA_FILE = "data.json"
DEFAULT_DATA = {
    "nicknames": {},
    "greeting_prefs": {},
    "silent_prefs": {},
    "memos": {},
    "model_prefs": {}
}

# =================================
#  記憶ファイルの読み書き機能 (安定版)
# =================================


def load_data() -> dict:
    """data.jsonからデータを安全に読み込む"""
    try:
        if os.path.exists(DATA_FILE):
            with open(DATA_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
    except (json.JSONDecodeError, OSError):
        pass
    return DEFAULT_DATA.copy()


def atomic_write_json(path: str, data: dict):
    """データが壊れないように安全に書き込む"""
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)


# =================================
#  Geminiに質問するための関数 (★★★ 安全設定を強化 ★★★)
# =================================


async def ask_gemini_async(message_text: str, user_memos: Dict[str, str],
                           model_name: str) -> str:
    """記憶情報と指定されたモデルを活用して、Geminiが返事を考える"""
    if not GEMINI_API_KEY:
        return "（エラー: GeminiのAPIキーが設定されてないみたい…）"

    def _run_sync() -> str:
        try:
            import google.generativeai as genai
            from google.generativeai.types import HarmCategory, HarmBlockThreshold

            genai.configure(api_key=GEMINI_API_KEY)

            memo_context = "ユーザーに関する以下の記憶情報を参考にして、会話をよりパーソナルなものにしてください。\n"
            if user_memos:
                for key, value in user_memos.items():
                    memo_context += f"- {key}: {value}\n"
            else:
                memo_context += "まだユーザーに関する記憶はありません。\n"

            system_instruction = (
                "あなたは『しかくうつ』という名前のキャラクターです。やさしくて、活発で、少しぽんこつな男の子として、"
                "フレンドリーかつ短い文章で返事をしてください。重要：顔文字は使わず、言葉だけで感情を表現してください。\n\n"
                f"{memo_context}")

            model = genai.GenerativeModel(
                model_name, system_instruction=system_instruction)

            GenerationConfig = getattr(getattr(genai, "types", genai),
                                       "GenerationConfig", None)
            cfg = GenerationConfig(candidate_count=1,
                                   max_output_tokens=150,
                                   temperature=0.8)

            # ★ 安全設定をより確実なリスト形式に変更
            safety_settings = [
                {
                    "category": HarmCategory.HARM_CATEGORY_HARASSMENT,
                    "threshold": HarmBlockThreshold.BLOCK_NONE
                },
                {
                    "category": HarmCategory.HARM_CATEGORY_HATE_SPEECH,
                    "threshold": HarmBlockThreshold.BLOCK_NONE
                },
                {
                    "category": HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT,
                    "threshold": HarmBlockThreshold.BLOCK_NONE
                },
                {
                    "category": HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,
                    "threshold": HarmBlockThreshold.BLOCK_NONE
                },
            ]

            response = model.generate_content(message_text,
                                              generation_config=cfg,
                                              safety_settings=safety_settings)

            if response.parts:
                return response.text
            else:
                print(
                    f"[Gemini] Response blocked or empty. Finish reason: {response.candidates[0].finish_reason}"
                )
                return "（ごめん！その言葉、なんだかうまく話せないみたい…別の言い方で聞いてみてくれる？）"

        except Exception as e:
            print(f"[Gemini] error: {e}")
            return "（エラー: 今、頭の中がぐるぐるしてる…）"

    return await asyncio.to_thread(_run_sync)


# =================================
#  Open‑Meteo API（キー不要）のラッパ
# =================================


async def geocode_place(
        session: aiohttp.ClientSession,
        place: str,
        language: str = "ja") -> Optional[Tuple[float, float, str]]:
    """地名→緯度経度。見つからなければNone。"""
    try:
        url = "https://geocoding-api.open-meteo.com/v1/search"
        params = {"name": place, "count": 1, "language": language}
        async with session.get(url, params=params, timeout=10) as r:
            if r.status != 200:
                return None
            data = await r.json()
        results = data.get("results") or []
        if not results:
            return None
        item = results[0]
        lat = float(item.get("latitude"))
        lon = float(item.get("longitude"))
        label = ", ".join(
            x for x in
            [item.get("name"),
             item.get("admin1"),
             item.get("country")] if x)
        return (lat, lon, label)
    except Exception as e:
        print(f"[Weather] geocode error: {e}")
        return None


async def fetch_forecast(session: aiohttp.ClientSession,
                         lat: float,
                         lon: float,
                         tzname: str = "Asia/Tokyo") -> Optional[dict]:
    """日次の天気（今日/明日）を取得"""
    try:
        url = "https://api.open-meteo.com/v1/forecast"
        params = {
            "latitude": lat,
            "longitude": lon,
            "daily":
            "weathercode,temperature_2m_max,temperature_2m_min,precipitation_sum,wind_speed_10m_max",
            "timezone": tzname,
        }
        async with session.get(url, params=params, timeout=10) as r:
            if r.status != 200:
                return None
            return await r.json()
    except Exception as e:
        print(f"[Weather] forecast error: {e}")
        return None


WMO_CODE = {
    0: ("快晴", "☀"),
    1: ("晴れ", "🌤"),
    2: ("薄曇り", "⛅"),
    3: ("曇り", "☁"),
    45: ("霧", "🌫"),
    48: ("霧氷", "🌫"),
    51: ("霧雨(弱)", "🌦"),
    53: ("霧雨(中)", "🌦"),
    55: ("霧雨(強)", "🌦"),
    61: ("雨(弱)", "🌧"),
    63: ("雨(中)", "🌧"),
    65: ("雨(強)", "🌧"),
    66: ("凍雨(弱)", "🌧"),
    67: ("凍雨(強)", "🌧"),
    71: ("雪(弱)", "🌨"),
    73: ("雪(中)", "🌨"),
    75: ("雪(強)", "🌨"),
    77: ("雪粒", "🌨"),
    80: ("にわか雨(弱)", "🌦"),
    81: ("にわか雨(中)", "🌦"),
    82: ("にわか雨(強)", "⛈"),
    85: ("にわか雪(弱)", "🌨"),
    86: ("にわか雪(強)", "🌨"),
    95: ("雷雨(弱〜中)", "⛈"),
    96: ("雷雨+雹(弱)", "⛈"),
    99: ("雷雨+雹(強)", "⛈"),
}


def wmo_to_text(code: int) -> Tuple[str, str]:
    label, emoji = WMO_CODE.get(int(code), ("不明", "❔"))
    return label, emoji


# =================================
#  しかくうつBotのクラス（設計図）
# =================================


class ShikakuUtsu(discord.Client):

    def __init__(self, *, intents: discord.Intents):
        super().__init__(intents=intents)
        self.tree = app_commands.CommandTree(self)

        self.bot_data = load_data()
        self.nicknames: Dict[str, str] = self.bot_data.get("nicknames", {})
        self.greeting_prefs: Dict[str, bool] = self.bot_data.get(
            "greeting_prefs", {})
        self.silent_prefs: Dict[str,
                                bool] = self.bot_data.get("silent_prefs", {})
        self.memos: Dict[str, Dict[str, str]] = self.bot_data.get("memos", {})
        self.model_prefs: Dict[str, str] = self.bot_data.get("model_prefs", {})

        self.chat_switch: Dict[int, bool] = {}
        self.active_timers: list[dict] = []
        self.default_chat_on = True
        self.session: Optional[aiohttp.ClientSession] = None

    def save_bot_data(self):
        self.bot_data["nicknames"] = self.nicknames
        self.bot_data["greeting_prefs"] = self.greeting_prefs
        self.bot_data["silent_prefs"] = self.silent_prefs
        self.bot_data["memos"] = self.memos
        self.bot_data["model_prefs"] = self.model_prefs
        try:
            atomic_write_json(DATA_FILE, self.bot_data)
        except OSError as e:
            print(f"[Storage] failed to save data: {e}")

    async def setup_hook(self):
        self.session = aiohttp.ClientSession()
        self.check_timers.start()
        try:
            if GUILD_ID:
                guild_obj = discord.Object(id=int(GUILD_ID))
                self.tree.copy_global_to(guild=guild_obj)
                synced = await self.tree.sync(guild=guild_obj)
                print(
                    f"[Slash] Synced {len(synced)} commands to guild {GUILD_ID}"
                )
            else:
                synced = await self.tree.sync()
                print(f"[Slash] Synced {len(synced)} global commands")
        except Exception as e:
            print(f"[Slash] sync failed: {e}")

    async def close(self):
        try:
            if self.session:
                await self.session.close()
        finally:
            await super().close()

    async def on_ready(self):
        print(f"🔥 Logged in as {self.user} ({self.user.id})")

    @tasks.loop(seconds=5.0)
    async def check_timers(self):
        now = datetime.datetime.now(datetime.timezone.utc)
        due = [
            t for t in list(self.active_timers)
            if t.get("time") and t["time"] <= now
        ]
        for t in due:
            channel = self.get_channel(t.get("channel_id"))
            if isinstance(channel, (discord.TextChannel, discord.Thread)):
                user = self.get_user(t.get("user_id"))
                mention = user.mention if user else f"<@{t.get('user_id')}>"
                try:
                    await channel.send(f"{mention} {t.get('message', '')}")
                except Exception as e:
                    print(f"[Timer] send failed: {e}")
            try:
                self.active_timers.remove(t)
            except ValueError:
                pass

    @check_timers.before_loop
    async def before_check_timers(self):
        await self.wait_until_ready()

    async def on_message(self, message: discord.Message):
        if message.author.bot or not message.guild:
            return
        if not self.chat_switch.get(message.channel.id, self.default_chat_on):
            return

        text = (message.content or "").strip()
        if not text:
            return

        author_id_str = str(message.author.id)

        async def reply_helper(reply_text: str):
            final_reply = reply_text
            if self.greeting_prefs.get(author_id_str, False):
                nickname = self.nicknames.get(author_id_str,
                                              message.author.display_name)
                final_reply = f"{nickname}、{final_reply}"
            if self.silent_prefs.get(author_id_str, False):
                if not final_reply.startswith("@silent\n"):
                    final_reply = f"@silent\n{final_reply}"
            try:
                await message.channel.send(final_reply)
            except discord.Forbidden:
                print("[Send] Forbidden: Missing permissions in channel")
            except Exception as e:
                print(f"[Send] Failed: {e}")

        async with message.channel.typing():
            await asyncio.sleep(random.uniform(0.25, 0.6))

        if text in {
                self.user.name,
                getattr(self.user, "display_name", self.user.name)
        }:
            await reply_helper("はい")
            return

        dict_map = {
            "カツ丼": "いつだってたべたい！だいすき！",
            "おやつ": "今日はなに？",
            "アイス": "どの味にする？",
            "リンキン・パーク": "心が叫んでる！",
            "レイジ": "ゲリララジオ！",
            "マリリン・マンソン": "かっこいい、あくのカリスマ",
            "ニルヴァーナ": "なんだか叫びたい気分かも～",
            "カート・コバーン": "とってもリスペクトしてるよ",
            "ロック": "じぶんは、じぶんだ！",
            "ギター": "今日はどの曲弾く？",
            "へっぽこ": "へっぽこじゃない！",
            "こんにちは": "ちわ！",
            "やっほ～": "ちわ！",
            "やあ": "ちわ！",
            "hi": "ちわ！",
            "ちわ！": "ちわ！",
            "ㄘʓ‎〜": "ㄘʓ‎〜",
        }
        for k, v in dict_map.items():
            if k in text:
                await reply_helper(v)
                return

        if random.random() < 0.30:
            await reply_helper(
                random.choice(
                    ["うんうん", "なるほど～", "そっかそっか", "へぇ！", "はぇ～～", "メモした！"]))
            return

        user_memos = self.memos.get(author_id_str, {})
        model_name = self.model_prefs.get(author_id_str, "gemini-1.5-flash")
        reply_text = await ask_gemini_async(text, user_memos, model_name)
        await reply_helper(reply_text)


# =================================
#  Botを実際に起動する処理
# =================================

client = ShikakuUtsu(intents=intents)

# =================================
#  スラッシュコマンドの定義
# =================================


@client.tree.command(name="ping", description="生存確認！")
async def ping_cmd(interaction: discord.Interaction):
    await interaction.response.send_message("Pong! 生きてるよ～！")


@client.tree.command(name="chat", description="おしゃべりの on/off")
@app_commands.describe(mode="on / off")
async def chat_cmd(interaction: discord.Interaction, mode: Literal["on",
                                                                   "off"]):
    on = (mode == "on")
    client.chat_switch[interaction.channel.id] = on
    label = "おしゃべり" if on else "おしゃべりお休み"
    await interaction.response.send_message(f"このチャンネルは **{label}** にするね",
                                            ephemeral=True)


@client.tree.command(name="set_name", description="あなたの呼び名を覚えるよ")
@app_commands.describe(nickname="覚えてほしい名前を教えて")
async def set_name_cmd(interaction: discord.Interaction, nickname: str):
    client.nicknames[str(interaction.user.id)] = nickname
    client.save_bot_data()
    await interaction.response.send_message(f"わかった！これからは『{nickname}』って呼ぶね。",
                                            ephemeral=True)


@client.tree.command(name="set_greeting", description="返事の最初に名前を呼ぶか設定するよ")
@app_commands.describe(mode="on / off")
async def set_greeting_cmd(interaction: discord.Interaction,
                           mode: Literal["on", "off"]):
    on = (mode == "on")
    client.greeting_prefs[str(interaction.user.id)] = on
    client.save_bot_data()
    status = "これからは名前を呼ぶね！" if on else "わかった、名前は呼ばないようにするね。"
    await interaction.response.send_message(status, ephemeral=True)


@client.tree.command(name="set_silent", description="あなたへの返信をサイレントモードにするよ")
@app_commands.describe(mode="on / off")
async def set_silent_cmd(interaction: discord.Interaction,
                         mode: Literal["on", "off"]):
    on = (mode == "on")
    client.silent_prefs[str(interaction.user.id)] = on
    client.save_bot_data()
    status = "返信をサイレントモードにしたよ。通知は飛ばないはず！" if on else "サイレントモードを解除したよ。"
    await interaction.response.send_message(status, ephemeral=True)


@client.tree.command(name="set_model", description="会話に使うAIモデルを変更するよ")
@app_commands.describe(model="モデルを選んでね (flashは速くて安い, proは高性能)")
async def set_model_cmd(interaction: discord.Interaction,
                        model: Literal["gemini-1.5-flash", "gemini-1.5-pro"]):
    user_id = str(interaction.user.id)
    client.model_prefs[user_id] = model
    client.save_bot_data()
    await interaction.response.send_message(f"わかった！これからの会話は `{model}` を使うね！",
                                            ephemeral=True)


@client.tree.command(name="reset_my_settings",
                     description="僕が覚えたあなたの設定をリセットするよ")
async def reset_my_settings_cmd(interaction: discord.Interaction):
    user_id = str(interaction.user.id)
    client.nicknames.pop(user_id, None)
    client.greeting_prefs.pop(user_id, None)
    client.silent_prefs.pop(user_id, None)
    client.model_prefs.pop(user_id, None)
    client.save_bot_data()
    await interaction.response.send_message("わかった！あなたの設定を全部忘れちゃった！",
                                            ephemeral=True)


@client.tree.command(name="my_status", description="僕が覚えているあなたの設定を表示するよ")
async def my_status_cmd(interaction: discord.Interaction):
    user_id = str(interaction.user.id)
    nickname = client.nicknames.get(user_id, "まだ覚えてないよ")
    greeting_on = client.greeting_prefs.get(user_id, False)
    silent_on = client.silent_prefs.get(user_id, False)
    model_name = client.model_prefs.get(user_id, "gemini-1.5-flash")
    embed = discord.Embed(title=f"{interaction.user.display_name}さんの設定",
                          color=discord.Color.green())
    embed.add_field(name="呼び名", value=nickname, inline=False)
    embed.add_field(name="挨拶で名前を呼ぶ",
                    value=("オン" if greeting_on else "オフ"),
                    inline=False)
    embed.add_field(name="サイレントモード",
                    value=("オン" if silent_on else "オフ"),
                    inline=False)
    embed.add_field(name="使用モデル", value=f"`{model_name}`", inline=False)
    await interaction.response.send_message(embed=embed, ephemeral=True)


# ===== メモ機能 =====

memo_group = app_commands.Group(name="memo",
                                description="しかくうつに色々なことを覚えさせるコマンド")


@memo_group.command(name="add", description="新しいことを覚えるよ")
@app_commands.describe(keyword="キーワード", content="覚えてほしい内容")
async def memo_add(interaction: discord.Interaction, keyword: str,
                   content: str):
    user_id = str(interaction.user.id)
    if user_id not in client.memos:
        client.memos[user_id] = {}
    client.memos[user_id][keyword] = content
    client.save_bot_data()
    await interaction.response.send_message(f"「{keyword}」は「{content}」だね。覚えた！",
                                            ephemeral=True)


@memo_group.command(name="show", description="覚えていることを教えてくれるよ")
@app_commands.describe(keyword="知りたいキーワード")
async def memo_show(interaction: discord.Interaction, keyword: str):
    user_id = str(interaction.user.id)
    content = client.memos.get(user_id, {}).get(keyword)
    if content:
        await interaction.response.send_message(f"「{keyword}」は「{content}」だよ！",
                                                ephemeral=True)
    else:
        await interaction.response.send_message(
            f"ごめん、「{keyword}」についてはまだ知らないや…", ephemeral=True)


@memo_group.command(name="list", description="覚えているキーワードの一覧を表示するよ")
async def memo_list(interaction: discord.Interaction):
    user_id = str(interaction.user.id)
    user_memos = client.memos.get(user_id, {})
    if not user_memos:
        await interaction.response.send_message("まだ何も覚えてないよ！", ephemeral=True)
        return
    embed = discord.Embed(title=f"{interaction.user.display_name}さんから教わったこと",
                          color=discord.Color.purple())
    memo_list_text = "\n".join(f"- {key}" for key in user_memos.keys())
    embed.description = memo_list_text
    await interaction.response.send_message(embed=embed, ephemeral=True)


@memo_group.command(name="forget", description="覚えたことを忘れさせるよ")
@app_commands.describe(keyword="忘れてほしいキーワード")
async def memo_forget(interaction: discord.Interaction, keyword: str):
    user_id = str(interaction.user.id)
    if user_id in client.memos and keyword in client.memos[user_id]:
        del client.memos[user_id][keyword]
        client.save_bot_data()
        await interaction.response.send_message(f"「{keyword}」のこと、忘れちゃった！",
                                                ephemeral=True)
    else:
        await interaction.response.send_message(
            f"ごめん、「{keyword}」については元から知らなかったみたい…", ephemeral=True)


client.tree.add_command(memo_group)

# ===== 管理系 =====


@client.tree.command(name="all_erase", description="しかくうつの記憶をすべて消す（管理者専用）")
async def all_erase_cmd(interaction: discord.Interaction):
    is_owner = interaction.guild and (interaction.user.id
                                      == interaction.guild.owner_id)
    perms = interaction.user.guild_permissions if interaction.guild else None
    if not (is_owner or (perms and perms.manage_guild)):
        await interaction.response.send_message("このコマンドは管理者専用だよ！",
                                                ephemeral=True)
        return
    client.nicknames.clear()
    client.greeting_prefs.clear()
    client.silent_prefs.clear()
    client.memos.clear()
    client.model_prefs.clear()
    client.save_bot_data()
    await interaction.response.send_message("オールクリア！まっさらにしたよ。", ephemeral=True)


@client.tree.command(name="memory_read", description="しかくうつが記憶している情報を開示するよ")
async def memory_read_cmd(interaction: discord.Interaction):
    is_owner = interaction.guild and (interaction.user.id
                                      == interaction.guild.owner_id)
    perms = interaction.user.guild_permissions if interaction.guild else None
    can_view_all = bool(is_owner or (perms and perms.manage_guild))
    data = {"scope": "all" if can_view_all else "self"}
    uid = str(interaction.user.id)
    if can_view_all:
        data.update(client.bot_data)
    else:
        data.update({
            "nicknames": {
                uid: client.nicknames.get(uid)
            } if uid in client.nicknames else {},
            "greeting_prefs": {
                uid: client.greeting_prefs.get(uid, False)
            },
            "silent_prefs": {
                uid: client.silent_prefs.get(uid, False)
            },
            "memos": {
                uid: client.memos.get(uid, {})
            } if uid in client.memos else {},
            "model_prefs": {
                uid: client.model_prefs.get(uid)
            } if uid in client.model_prefs else {}
        })
    text = json.dumps(data, ensure_ascii=False, indent=2)
    if len(text) > 1800:
        fp = io.BytesIO(text.encode("utf-8"))
        file = discord.File(fp, filename="shikaku_memory.json")
        await interaction.response.send_message(content="記憶データを添付したよ。",
                                                file=file,
                                                ephemeral=True)
    else:
        await interaction.response.send_message(f"```json\n{text}\n```",
                                                ephemeral=True)


# ===== 便利コマンド =====


@client.tree.command(name="help", description="しかくうつの使い方を表示するよ")
async def help_cmd(interaction: discord.Interaction):
    lines = [
        "**しかくうつ — コマンド一覧**",
        "## しかくうつは試験運用中です。",
        "・/ping — 生存確認",
        "・/chat on|off — このチャンネルでの自動おしゃべり切替（その人だけに見える）",
        "・/set_name <nickname> — あなたの呼び名を登録（その人だけに見える）",
        "・/set_greeting on|off — 返信の最初に名前を呼ぶか（その人だけに見える）",
        "・/set_silent on|off — あなたへの返信をサイレント（@silent\\n付与、個人のみ表示）",
        "・/set_model <model> — 会話に使うAIモデルを変更（その人だけに見える）",
        "・/my_status — あなたの現在設定を表示（その人だけに見える）",
        "・/reset_my_settings — あなたの設定をリセット（その人だけに見える）",
        "・/memo add|show|list|forget — 個人メモ（その人だけに見える）",
        "・/memory_read — 記憶を開示（管理者は全体、一般は自分のみ・個人表示）",
        "・/all_erase — **全記憶消去（管理者専用・個人表示）**",
        "・/timer <hours> <minutes> [message] — 指定の時間後に通知（全体）",
        "・/alarm <hour> <minute> [message] — 指定時刻に1回通知（全体）",
        "・/omikuji — おみくじ（全体）",
        "・/weather <place> — 天気（全体）",
        "・/dice <count> <sides> [plus] — ダイス（全体）",
        "・/rps <hand> — じゃんけん（全体）",
        "・/which <options> [n] [unique] — 候補から選ぶ（全体）",
        "-# ⓘしかくうつのGemini Apiは有料コンテンツです。みんなにしかくうつと遊んでほしいので、頑張って資金繰りをしています。",
        "-#  強制ではありませんが、もし『しかくうつ』が好きだよっていう人は丸の鬱に資金をぶち投げることができます。詳しくは本人まで。",
    ]
    await interaction.response.send_message("\n".join(lines))


@client.tree.command(name="timer", description="○時間○分後に通知するよ")
@app_commands.describe(hours="0以上の時間", minutes="0-59の分", message="任意: メッセージ")
async def timer_cmd(interaction: discord.Interaction,
                    hours: app_commands.Range[int, 0, None],
                    minutes: app_commands.Range[int, 0, 59],
                    message: Optional[str] = None):
    if hours == 0 and minutes == 0:
        await interaction.response.send_message("0時間0分は設定できないよ。",
                                                ephemeral=True)
        return
    delta = datetime.timedelta(hours=int(hours), minutes=int(minutes))
    target_utc = datetime.datetime.now(datetime.timezone.utc) + delta
    client.active_timers.append({
        "time": target_utc,
        "channel_id": interaction.channel.id,
        "user_id": interaction.user.id,
        "message": message or "時間だよ！",
    })
    await interaction.response.send_message(
        f"タイマーOK。{hours}時間{minutes}分後に知らせるね。")


@client.tree.command(name="alarm", description="今日の何時何分に一度だけ通知するよ（過ぎてたら翌日に）")
@app_commands.describe(hour="0-23の時", minute="0-59の分", message="任意: メッセージ")
async def alarm_cmd(interaction: discord.Interaction,
                    hour: app_commands.Range[int, 0, 23],
                    minute: app_commands.Range[int, 0, 59],
                    message: Optional[str] = None):
    now_local = datetime.datetime.now(TZ)
    try:
        target_local = now_local.replace(hour=int(hour),
                                         minute=int(minute),
                                         second=0,
                                         microsecond=0)
    except ValueError:
        await interaction.response.send_message("その時刻はおかしいかも。", ephemeral=True)
        return
    if target_local <= now_local:
        target_local += datetime.timedelta(days=1)
    target_utc = target_local.astimezone(datetime.timezone.utc)
    client.active_timers.append({
        "time": target_utc,
        "channel_id": interaction.channel.id,
        "user_id": interaction.user.id,
        "message": message or "起きて！",
    })
    disp_day = target_local.strftime("%m/%d")
    await interaction.response.send_message(
        f"アラームOK。{disp_day} {hour:02d}:{minute:02d} に知らせるね。")


@client.tree.command(name="omikuji", description="おみくじを引くよ（大吉〜大凶）")
async def omikuji_cmd(interaction: discord.Interaction):
    results = [
        {
            "name": "大吉",
            "message": "さすがだね！今日は無敵！何をやってもうまくいく最高の一日！",
            "advice": "新しいことに挑戦してみて！",
            "lucky_item": "光るもの",
            "color": discord.Color.gold(),
            "weight": 5
        },
        {
            "name": "中吉",
            "message": "いいかんじ！おいしいものでもたべよう！",
            "advice": "周りの人に親切にすると吉。",
            "lucky_item": "お気に入りの音楽",
            "color": discord.Color.red(),
            "weight": 15
        },
        {
            "name": "小吉",
            "message": "プチいいこと、あるかもしれないね。",
            "advice": "身の回りの整理整頓を心がけて。",
            "lucky_item": "植物",
            "color": discord.Color.orange(),
            "weight": 20
        },
        {
            "name": "吉",
            "message": "何もない日もいいね。平和が一番！",
            "advice": "のんびり過ごすのが良さそう。",
            "lucky_item": "読みかけの本",
            "color": discord.Color.green(),
            "weight": 25
        },
        {
            "name": "末吉",
            "message": "ありゃ？これは結んで帰ろうね。",
            "advice": "忘れ物に注意！確認を怠らないで。",
            "lucky_item": "ハンカチ",
            "color": discord.Color.blue(),
            "weight": 20
        },
        {
            "name": "凶",
            "message": "うわわ、今日はツイてないかも…",
            "advice": "慎重に行動しよう。焦りは禁物。",
            "lucky_item": "温かいお茶",
            "color": discord.Color.dark_grey(),
            "weight": 10
        },
        {
            "name": "大凶",
            "message": "今すぐお払いに行こう……😭",
            "advice": "今日は無理せず、早めに休むのが一番。",
            "lucky_item": "お守り",
            # Color.black() は無いので from_rgb(0,0,0) か dark_grey() に置換
            "color": discord.Color.dark_grey(),
            # もし完全な黒が良ければ下記のどちらかに変更：
            # "color": discord.Color.from_rgb(0, 0, 0),
            # "color": discord.Color(0x000000),
            "weight": 5
        },
    ]

    population = [r for r in results]
    weights = [r["weight"] for r in results]
    pick = random.choices(population, weights=weights, k=1)[0]

    embed = discord.Embed(title=f"今日の運勢は…… **{pick['name']}**！",
                          description=pick['message'],
                          color=pick['color'])
    embed.add_field(name="アドバイス", value=pick['advice'], inline=False)
    embed.add_field(name="ラッキーアイテム", value=pick['lucky_item'], inline=False)
    embed.set_footer(text=f"{interaction.user.display_name}さんの今日の運勢")

    await interaction.response.send_message(embed=embed)


# ===== /weather =====


@client.tree.command(name="weather",
                     description="地名を入れると、今日と明日の天気を教えるよ（Open‑Meteo）")
@app_commands.describe(place="例: 名古屋, Tokyo, Osaka, Sapporo など")
async def weather_cmd(interaction: discord.Interaction, place: str):
    await interaction.response.defer()  # 3秒ルール回避
    if client.session is None:
        await interaction.followup.send("内部エラー: HTTPセッション未初期化", ephemeral=True)
        return

    geo = await geocode_place(client.session, place, language="ja")
    if not geo:
        await interaction.followup.send(f"『{place}』が見つからなかったよ……別の言い方で試してみて！",
                                        ephemeral=True)
        return

    lat, lon, label = geo
    data = await fetch_forecast(client.session, lat, lon, tzname=DEFAULT_TZ)
    if not data:
        await interaction.followup.send("予報の取得に失敗しちゃった。ちょっと時間をおいてみて。",
                                        ephemeral=True)
        return

    daily = data.get("daily", {})
    times = daily.get("time", [])
    wcodes = daily.get("weathercode", [])
    tmax = daily.get("temperature_2m_max", [])
    tmin = daily.get("temperature_2m_min", [])
    prcp = daily.get("precipitation_sum", [])
    wind = daily.get("wind_speed_10m_max", [])

    def row(idx: int) -> str:
        if idx >= len(times):
            return "—"
        d = times[idx]
        wc = wmo_to_text(wcodes[idx])[0] if idx < len(wcodes) else "—"
        em = wmo_to_text(wcodes[idx])[1] if idx < len(wcodes) else ""
        hi = f"{tmax[idx]:.0f}°C" if idx < len(tmax) else "—"
        lo = f"{tmin[idx]:.0f}°C" if idx < len(tmin) else "—"
        pp = f"{prcp[idx]:.1f}mm" if idx < len(prcp) else "—"
        wd = f"{wind[idx]:.0f}m/s" if idx < len(wind) else "—"
        return f"{d}  {em} {wc}  最高 {hi} / 最低 {lo}  降水 {pp}  風 {wd}"

    embed = discord.Embed(title=f"{label} の天気（{DEFAULT_TZ}）",
                          color=discord.Color.blue())
    embed.add_field(name="今日", value=row(0), inline=False)
    if len(times) > 1:
        embed.add_field(name="明日", value=row(1), inline=False)
    embed.set_footer(text="Powered by Open‑Meteo")

    await interaction.followup.send(embed=embed)


# ===== /dice =====


@client.tree.command(name="dice",
                     description="ダイスを振るよ：/dice <個数> <面数> [加算] 例) /dice 2 6 3")
@app_commands.describe(count="振る個数（1-100）",
                       sides="何面ダイスか（2-1000）",
                       plus="合計に足す修正値（任意、-999〜999）")
async def dice_cmd(interaction: discord.Interaction,
                   count: app_commands.Range[int, 1, 100],
                   sides: app_commands.Range[int, 2, 1000],
                   plus: Optional[app_commands.Range[int, -999, 999]] = 0):
    rolls = [random.randint(1, sides) for _ in range(count)]
    total = sum(rolls) + int(plus or 0)

    # 長すぎる時は途中省略
    if len(rolls) > 50:
        head = ", ".join(map(str, rolls[:25]))
        tail = ", ".join(map(str, rolls[-25:]))
        rolls_text = f"{head}, …, {tail}"
    else:
        rolls_text = ", ".join(map(str, rolls))

    mod_text = f" + {plus}" if plus and plus > 0 else (
        f" - {abs(plus)}" if plus and plus < 0 else "")
    embed = discord.Embed(title="🎲 ダイス結果", color=discord.Color.teal())
    embed.add_field(name="式", value=f"{count}d{sides}{mod_text}", inline=True)
    embed.add_field(name="合計", value=str(total), inline=True)
    embed.add_field(name="出目", value=rolls_text or "(なし)", inline=False)
    await interaction.response.send_message(embed=embed)


# ===== /rps =====


@client.tree.command(name="rps", description="じゃんけん：グー/チョキ/パーで勝負！")
@app_commands.describe(hand="グー / チョキ / パー")
async def rps_cmd(interaction: discord.Interaction, hand: Literal["グー", "チョキ",
                                                                  "パー"]):
    bot_map = {"グー": "✊", "チョキ": "✌️", "パー": "🖐️"}
    you = hand
    bot = random.choice(list(bot_map.keys()))

    # 勝敗判定
    rules = {"グー": "チョキ", "チョキ": "パー", "パー": "グー"}
    if you == bot:
        result = "引き分け！もう一戦いく？"
    elif rules[you] == bot:
        result = "勝ち！やるじゃん！"
    else:
        result = "負け！ドンマイ、次は勝とう。"

    embed = discord.Embed(title="✊✌️🖐️ じゃんけん", color=discord.Color.blurple())
    embed.add_field(name="あなた", value=f"{bot_map[you]} {you}")
    embed.add_field(name="しかくうつ", value=f"{bot_map[bot]} {bot}")
    embed.add_field(name="結果", value=result, inline=False)
    await interaction.response.send_message(embed=embed)


# ===== /which =====


@client.tree.command(name="which", description="候補からランダムに選ぶよ")
@app_commands.describe(options="候補を区切って入力（, や | や 空白でOK）",
                       n="選ぶ数（1-10）",
                       unique="同じ候補を重複させない")
async def which_cmd(interaction: discord.Interaction,
                    options: str,
                    n: app_commands.Range[int, 1, 10] = 1,
                    unique: bool = True):
    # 区切り文字：カンマ、パイプ、スペース、日本語読点など
    items = [s.strip() for s in re.split(r"[|,、\s]+", options) if s.strip()]

    # 重複除去（順序維持）
    seen = set()
    dedup = []
    for x in items:
        if x not in seen:
            seen.add(x)
            dedup.append(x)

    if len(dedup) < 2:
        await interaction.response.send_message(
            "候補は2つ以上入れてね。例：/which ラーメン, カレー, 寿司", ephemeral=True)
        return

    if unique and n > len(dedup):
        await interaction.response.send_message("unique=真 のときは、選ぶ数は候補数以下にしてね。",
                                                ephemeral=True)
        return

    picks = (random.sample(dedup, k=n)
             if unique else [random.choice(dedup) for _ in range(n)])

    embed = discord.Embed(title="🎯 ランダムチョイス", color=discord.Color.green())
    embed.add_field(name="候補", value=", ".join(dedup)[:1000], inline=False)
    embed.add_field(name="選ばれたのは…", value=" / ".join(picks), inline=False)
    if not unique and n > 1:
        embed.set_footer(text="※ 重複ありで選んでいます")
    await interaction.response.send_message(embed=embed)


# ===== エラーハンドラ =====


@client.tree.error
async def on_app_command_error(interaction: discord.Interaction,
                               error: app_commands.AppCommandError):
    try:
        if isinstance(error, app_commands.CommandInvokeError):
            msg = f"実行中にエラー: {error.original}"
        elif isinstance(error, app_commands.TransformerError):
            msg = "引数の形式が正しくないみたい…"
        elif isinstance(error, app_commands.MissingPermissions):
            msg = "権限が足りないよ……"
        else:
            msg = f"エラー: {error}"
        if interaction.response.is_done():
            await interaction.followup.send(msg, ephemeral=True)
        else:
            await interaction.response.send_message(msg, ephemeral=True)
    except Exception as e:
        print(f"[SlashErrorHandler] failed to respond: {e}")


def main():
    if not TOKEN:
        print("エラー: DISCORD_TOKENが設定されていません。ホスト環境の環境変数を確認してください。")
        return
    try:
        client.run(TOKEN)
    except discord.LoginFailure:
        print("エラー: トークンが無効かも。再発行/コピーし直してね。")
    except Exception as e:
        print(f"[Run] Unexpected error: {e}")


if __name__ == "__main__":
    main()
