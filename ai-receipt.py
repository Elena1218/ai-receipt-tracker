# Python 內建的 sqlite3 模組，建立與操作本地資料庫
import sqlite3

# 解析 AI 回傳的 JSON 格式字串
import json

# 產生不重複的唯一識別碼 (UUID)
import uuid

# 取得當前的系統時間以記錄建立時間
from datetime import datetime

# 檢查本機圖片檔案是否存在
import os

# ollama 官方函式庫，呼叫本機執行的視覺多模態模型
import ollama

import re

#從 ollama 模組中匯入 Client（客戶端）類別
from ollama import Client

# 引入 SpeechRecognition 模組，用於麥克風收音與語音轉文字
import speech_recognition as sr

client = Client(host='http://172.17.24.130:11434')

# -----------------------------------------------------------------
# 步驟一：初始化資料庫與建立資料表
# -----------------------------------------------------------------

# 定義初始化資料庫的函式，預設資料庫檔名為 expense_test.db
def init_db(db_name="expense_test.db"):
    # 連線到指定的 SQLite 資料庫檔案；若檔案不存在，系統會自動在同目錄建立
    conn = sqlite3.connect(db_name)
    
    # 建立游標物件 (cursor)，用來向資料庫傳送並執行 SQL 指令
    cursor = conn.cursor()
    
    # 執行 SQL 語法建立交易資料表 (若已存在則不重複建立)
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS expenses (
        id TEXT PRIMARY KEY,
        transaction_date TEXT,             -- 交易日期
        category TEXT,                     -- 消費類別
        item_name TEXT,                    -- 商品名稱
        amount REAL,                       -- 交易實付金額 (浮點數)
        target_person TEXT,                -- 使用者標記
        channel TEXT,                      -- 消費通路
        raw_ai_response TEXT,              -- 保存 AI 辨識回傳的完整原始文字
        created_at TEXT                    -- 資料寫入資料庫的時間戳記
    )
    """)
    
    # 提交 (commit) 事務，將建立資料表的變更寫入硬碟
    conn.commit()
    
    # 關閉資料庫連線，釋放系統資源
    conn.close()


# -----------------------------------------------------------------
# 步驟二：呼叫 AI 視覺模型辨識交易明細
# -----------------------------------------------------------------

# 定義利用 AI 辨識圖片內容的函式，接收圖片路徑與模型名稱
def parse_receipt_with_ai(image_path: str, model_name: str = "llama3.2-vision") -> dict:
    # 檢查輸入的圖片檔案路徑是否真實存在於硬碟上
    if not os.path.exists(image_path):
        # 若找不到檔案，主動拋出 FileNotFoundError 例外錯誤並停止執行
        raise FileNotFoundError(f"找不到指定的圖片檔案: {image_path}")

    # 設計提示詞 (Prompt)，嚴格規範 AI 擷取的欄位與強制回傳純 JSON 格式
    prompt = """
    你是一個專業的記帳單據辨識助手。
    請檢視這張交易明細/發票/訂單截圖，並以繁體中文提取出以下資訊。
    請務必僅輸出合法的 JSON 字串，不要包含任何額外的解釋文字或 Markdown 區塊標記：
    {
        "transaction_date": "交易日期，格式 YYYY-MM-DD，若找不到請填寫今日日期",
        "category": "分類，如：餐飲、食品、網購、日用品、娛樂、其他",
        "item_name": "主要品項名稱或摘要",
        "amount": 實付總金額(純數字，不可包含貨幣符號或逗號),
        "target_person": "若明細中有指定給誰請填寫，預設填寫 '自己'",
        "channel": "通路名稱，例如：7-ELEVEN、全聯、賣貨便、一般門市"
    }
    """

    # 輸出終端機提示，告知使用者目前正在傳送圖片給 AI 進行分析
    print("正在使用 AI 模型辨識圖片內容，請稍候...")

    # 呼叫 Ollama 的 chat 介面，將提示詞與圖片檔案路徑傳入
    response = client.chat(
        # 指定模型
        model='qwen3-vl:8b',
        # 組裝對話訊息清單
        messages=[{
            # 設定對話角色為使用者 (user)
            "role": "user",
            # 設定使用者的文字要求
            "content": prompt,
            # 提供待辨識的圖片路徑清單
            "images": [image_path]
        }],
        options={
            "num_ctx": 8192  # 將上下文視窗從 4096 擴大至 8192，解決 token 超出問題
        }
    )

    # 從回傳物件中取得模型輸出的原始字串內容
    ai_text = response['message']['content'].strip()

    # 清除 Markdown JSON 代碼塊開頭標記 (例如 ```json 或 ```)
    if ai_text.startswith("```json"):
        # 去除開頭的 ```json
        ai_text = ai_text[7:]
    elif ai_text.startswith("```"):
        # 去除開頭的 ```
        ai_text = ai_text[3:]
    if ai_text.endswith("```"):
        # 去除結尾的 ```
        ai_text = ai_text[:-3]
    # .strip()清除頭尾多餘的換行與空白
    ai_text = ai_text.strip()

    # 嘗試將模型產生的文字轉換成 Python 字典
    try:
        # 使用 json.loads 將符合規格的 JSON 字串轉成 dict
        parsed_data = json.loads(ai_text)
    except json.JSONDecodeError:
        # 若模型回傳格式非合法 JSON，印出錯誤訊息並回傳預設結構以防程式崩潰
        print(f"JSON 解析失敗，模型原始輸出內容為：\n{ai_text}")
        parsed_data = {
            "transaction_date": datetime.now().strftime("%Y-%m-%d"),
            "category": "未分類",
            "item_name": "手動檢查項目 (AI格式錯誤)",
            "amount": 0.0,
            "target_person": "自己",
            "channel": "未知"
        }

    # 將模型的原始輸出文字一併存回字典，方便後續寫入資料庫核對
    parsed_data["raw_ai_response"] = ai_text

    # 回傳結構化整理後的明細字典
    return parsed_data


# -----------------------------------------------------------------
# 步驟三：將辨識後的資料寫入 SQLite 資料庫
# -----------------------------------------------------------------

# 定義儲存交易明細至資料庫的函式
def save_expense_to_db(data: dict, db_name="expense_test.db") -> str:
    # 建立與資料庫的連線
    conn = sqlite3.connect(db_name)
    
    # 建立游標物件
    cursor = conn.cursor()
    
    # 產生一組標準的 UUIDv4 字串作為此筆紀錄的主鍵 (Primary Key)
    record_id = str(uuid.uuid4())
    
    # 取得當前系統的 ISO 格式時間字串
    now_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # 準備插入資料的 SQL 指令，使用 ? 作為參數化佔位符防止 SQL Injection
    sql = """
    INSERT INTO expenses (
        id, transaction_date, category, item_name, 
        amount, target_person, channel, raw_ai_response, created_at
    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    """

    # 將欄位變數組裝成元組 (tuple)
    values = (
        record_id,                                           # 紀錄主鍵 ID
        data.get("transaction_date", datetime.now().strftime("%Y-%m-%d")),# 交易日期
        data.get("category", "未分類"),                       # 分類
        data.get("item_name", "未知品項"),                    # 品項名稱
        float(data.get("amount", 0.0)),                      # 金額 (確保轉為浮點數)
        data.get("target_person", "自己"),                    # 買給誰
        data.get("channel", "一般門市"),                      # 通路
        data.get("raw_ai_response", ""),                     # 原始輸出
        now_time                                             # 建立時間
    )

    # 執行 SQL 語法寫入資料
    cursor.execute(sql, values)
    
    # 提交資料庫變更
    conn.commit()
    
    # 關閉連線
    conn.close()
    
    # 回傳這筆新資料的 ID
    return record_id


# -----------------------------------------------------------------
# 步驟四：列印全部記帳紀錄
# -----------------------------------------------------------------

# 定義讀取並列印記帳紀錄的函式
def list_all_expenses(db_name="expense_test.db"):
    # 建立與資料庫的連線
    conn = sqlite3.connect(db_name)
    
    # 建立游標物件
    cursor = conn.cursor()
    
    # 執行 SQL 查詢：選取所有欄位，並依照建立時間由新到舊排序
    cursor.execute("SELECT id, transaction_date, category, item_name, amount, target_person, channel FROM expenses ORDER BY created_at DESC")
    
    # 抓取所有符合條件的查詢結果
    rows = cursor.fetchall()
    
    # 印出表格標題列
    print("\n" + "=" * 80)
    print(f"{'日期':<12} | {'分類':<8} | {'通路':<10} | {'對象':<6} | {'金額':<8} | {'品項名稱'}")
    print("-" * 80)
    
    # 遍歷每筆查詢紀錄
    for row in rows:
        # 解包各欄位變數 欄位取名 方便後續呼叫
        record_id, t_date, cat, item, amt, person, ch = row
        # 格式化輸出對齊
        print(f"{t_date:<12} | {cat:<8} | {ch:<10} | {person:<6} | {amt:<8.1f} | {item}")
        
    print("=" * 80 + "\n")
    
    # 關閉連線
    conn.close()


# -----------------------------------------------------------------
# 步驟五：資料分析與多條件查詢模組 (查某天、某地點、某對象)
# -----------------------------------------------------------------

def query_expenses(db_name="expense_test.db", date=None, channel=None, target_person=None):
    """根據使用者輸入的條件動態查詢交易明細"""
    # 建立與資料庫的連線
    conn = sqlite3.connect(db_name)
    # 建立游標物件
    cursor = conn.cursor()
    
    # 設定基礎 SQL 查詢語法，使用 1=1 方便後續動態拼接 AND 條件
    query = "SELECT transaction_date, category, channel, target_person, amount, item_name FROM expenses WHERE 1=1"
    # 建立存放查詢參數值的清單
    params = []

    # 若有指定查詢日期，拼接精確比對條件
    if date:
        query += " AND transaction_date = ?"
        params.append(date)
    # 若有指定通路或地點，拼接模糊比對條件 (LIKE)
    if channel:
        query += " AND channel LIKE ?"
        params.append(f"%{channel}%")
    # 若有指定對象，拼接精確比對條件
    if target_person:
        query += " AND target_person = ?"
        params.append(target_person)
        
    # 將查詢結果按照交易日期由新到舊排序
    query += " ORDER BY transaction_date DESC"
    
    # 執行帶有參數的 SQL 查詢
    cursor.execute(query, params)
    
    # 取得所有篩選後的資料
    rows = cursor.fetchall()
    
    # 印出查詢結果表頭
    print("\n" + "=" * 80)
    print(f"{'日期':<12} | {'分類':<8} | {'通路':<12} | {'對象':<6} | {'金額':<8} | {'品項名稱'}")
    print("-" * 80)
    
    # 初始化篩選金額加總變數
    total = 0.0
    
    # 逐筆遍歷並解包資料
    for row in rows:
        t_date, cat, ch, person, amt, item = row
        # 累加實付金額
        total += amt
        # 輸出對齊格式字串
        print(f"{t_date:<12} | {cat:<8} | {ch:<12} | {person:<6} | {amt:<8.1f} | {item}")
        
    print("-" * 80)
    # 印出筆數統計與加總金額
    print(f"查詢結果共 {len(rows)} 筆，總支出金額：{total:.1f} 元")
    print("=" * 80 + "\n")
    
    # 關閉連線
    conn.close()


# -----------------------------------------------------------------
# 步驟六：維度聚合統計模組 (依地點或對象分組加總)
# -----------------------------------------------------------------

# 定義維度統計函式，支援依據 'channel' 或 'target_person' 分組
def show_summary_by_dimension(db_name="expense_test.db", dimension="channel"):
    # 安全驗證：限制分組維度欄位，避免外部任意字串造成安全隱患
    if dimension not in ["channel", "target_person", "category"]:
        return
        
    # 建立連線
    conn = sqlite3.connect(db_name)
    
    # 建立游標物件
    cursor = conn.cursor()
    
    # 定義欄位對應的中文標題字典
    label_map = {"channel": "通路/地點", "target_person": "消費對象", "category": "消費類別"}
    
    # 執行 SQL GROUP BY 分組聚合查詢：計算各組筆數與金額加總
    sql = f"""
    SELECT {dimension}, COUNT(*), SUM(amount)
    FROM expenses
    GROUP BY {dimension}
    ORDER BY SUM(amount) DESC
    """
    cursor.execute(sql)
    
    # 取得聚合統計資料
    rows = cursor.fetchall()
    
    # 印出統計報表表頭
    print("\n" + "=" * 45)
    print(f"{label_map[dimension]:<15} | {'筆數':<6} | {'總花費金額'}")
    print("-" * 45)
    
    # 逐行列印分組結果
    for name, count, total in rows:
        display_name = name if name else "未標記"
        print(f"{display_name:<15} | {count:<6} | {total:<10.1f}")
        
    print("=" * 45 + "\n")
    
    # 關閉連線
    conn.close()

# -----------------------------------------------------------------
# 口說搜尋_功能 1：麥克風收音轉文字 (Speech-to-Text)
# -----------------------------------------------------------------
def record_voice_to_text() -> str:
    """
    透過麥克風收音，並利用 Google 繁體中文語音辨識 API 轉換成字串
    """
    recognizer = sr.Recognizer()
    
    # 啟用本機預設麥克風
    with sr.Microphone() as source:
        print("\n[語音收音] 請開始說話（例如：『查一下昨天買了什麼』或『上禮拜四媽媽花了多少錢』）...")
        
        # 動態調整環境噪音基準
        recognizer.adjust_for_ambient_noise(source, duration=0.8)
        
        try:
            # 開始錄音，限制最長傾聽等待時間為 5 秒，說話長度最長 10 秒
            audio = recognizer.listen(source, timeout=5, phrase_time_limit=10)
            print("[語音收音] 錄音結束，正在辨識語音內容...")
            
            # 使用 Google 語音辨識轉換為繁體中文
            text = recognizer.recognize_google(audio, language="zh-TW")
            print(f"[語音辨識結果] 「{text}」")
            return text
            
        except sr.WaitTimeoutError:
            print("[語音提示] 超過等待時間，未偵測到聲音。")
            return ""
        except sr.UnknownValueError:
            print("[語音提示] 無法清楚辨識語音內容，請再試一次。")
            return ""
        except sr.RequestError as e:
            print(f"[語音錯誤] 語音辨識服務請求失敗: {e}")
            return ""


# -----------------------------------------------------------------
# 結合「今日日期基準」的 Text-to-SQL 生成
# -----------------------------------------------------------------
def generate_sql_with_date_context(user_query: str, model_name: str = "gpt-oss:latest") -> str:
    # 取得系統當前日期與星期
    weekdays_zh = ["星期一", "星期二", "星期三", "星期四", "星期五", "星期六", "星期日"]
    now = datetime.now()
    today_str = now.strftime("%Y-%m-%d")
    weekday_str = weekdays_zh[now.weekday()]

    # 在 Prompt 中提供絕對時間錨點 (Anchor)
    prompt = f"""
    你是一個精通 SQLite 與時間邏輯推理的資料分析師。
    你請嚴格依照以下【三步驟】思考，**每個步驟只准執行一次，絕對禁止重複驗算或自我懷疑**：
    【時間基準】：今天是 {today_str} ({weekday_str})。
    【資料庫結構】：
    資料表：expenses
    欄位：
    - id (TEXT)
    - transaction_date (TEXT, 格式固定為 'YYYY-MM-DD')
    - category (TEXT, 如：餐飲、日用品、食品、網購)
    - item_name (TEXT, 商品名稱)
    - amount (REAL, 金額)
    - target_person (TEXT, 對象：'自己'、'媽媽')
    - channel (TEXT, 通路/地點：'全聯'、'沃爾瑪'、'全家便利商店')

    【思考步驟（單向執行，做完即結束）】：
    步驟 1：計算目標日期或區間（例如上個月用 LIKE '2026-10-%'，上禮拜五算成 YYYY-MM-DD）。算出來後直接進入步驟 2，不准重算。
    步驟 2：決定 SELECT 欄位與 WHERE 條件（文字比對用 LIKE '%關鍵字%'）。
    步驟 3：立刻停止思考，直接輸出最終 SELECT 語句，不要任何標記或說明。

    使用者問題：{user_query}
    生成的 SQL 語法：
    """

    response = client.chat(
        model=model_name,
        messages=[{"role": "user", "content": prompt}],
        options={
            "temperature": 0.1,    # 保留微幅彈性讓推理更順暢
            "num_predict": 2048,   # 給足 2048 tokens，讓模型先完成 thinking 再輸出 content
            "repeat_penalty": 1.15,   # 抑制模型在思考中反覆鬼打牆
            "num_ctx": 8192
        }
    )
    
    # 從回傳字典中安全取得模型的內部思考過程 ('thinking' 欄位)；若模型沒有回傳此欄位，則預設為空字串 ""
    thinking_process = response['message'].get('thinking', '')
    
    # 從回傳字典中安全取得模型的正式回答字串 ('content' 欄位)；若沒有則預設為空字串 ""
    raw_response = response['message'].get('content', '')
    
    # 判斷思考過程是否有內容；若有，印出來讓開發者確認 AI 推算日曆與條件的邏輯是否正確
    if thinking_process:
        print(f"\n======== [DEBUG] AI 思考推導過程 ========\n{thinking_process}\n=========================================\n")
        
    # 使用 repr() 印出正式回覆的原始字串；repr() 會把換行符號(\n)或空格原形畢露，方便排查是否為空字串 ''
    print(f"======== [DEBUG] AI 正式回覆內容 ========\n{repr(raw_response)}\n=========================================\n")

    # 對原始回答呼叫 .strip()：移除字串「最開頭」與「最結尾」的所有空白字元、換行符號(\n)與定位字元(\t)
    sql_text = raw_response.strip()

    # 若清理完後發現根本是空字串，直接回傳 "" 交給後續的安全機制攔截，避免多餘的正規表達式運算
    if not sql_text:
        return ""

    # 使用正規表達式清理開頭的 Markdown 程式碼區塊標記 (例如 ```sql 或 ```SQL 或 ```)
    # ^ 代表字串開頭；(?:sql)? 代表可有可無的 sql 字樣；\s* 代表後面的換行或空格；flags=re.IGNORECASE 代表不分大小寫
    sql_text = re.sub(r"^```(?:sql)?\s*", "", sql_text, flags=re.IGNORECASE)
    
    # 清理結尾的 Markdown 程式碼區塊標記 (```)
    # \s* 代表前面的換行或空格；$ 代表字串結尾
    sql_text = re.sub(r"\s*```$", "", sql_text)

    # 進階保險擷取：用正規表達式尋找以 SELECT 開頭、直到整段結束（或分號結尾）的 SQL 語句
    # 作用是防止模型在 SQL 前面多講一句「好的，這是一段查詢語法：」導致後面程式判斷開頭不是 SELECT
    match = re.search(r"(SELECT[\s\S]+)", sql_text, re.IGNORECASE)
    
    # 如果有成功在回答中找到 SELECT 語句
    if match:
        # match.group(1) 取出括號內配對到的完整 SQL 語句，並再次用 .strip() 確保頭尾沒有殘留的空白或換行
        return match.group(1).strip()

    # 若沒配對到標準 SELECT 開頭，仍將清理過頭尾空白的字串回傳，交由外層的安全檢查機制處理
    return sql_text.strip()


# -----------------------------------------------------------------
# 執行查詢並呈現
# -----------------------------------------------------------------
def execute_natural_query(user_query: str, db_name="expense_test.db"):
    """執行大白話/語音辨識出的自然語言查詢"""
    if not user_query:
        return

    print(f"\n[AI 理解中] 正在分析問題：'{user_query}' ...")
    sql = generate_sql_with_date_context(user_query)
    print(f"[推算生成的 SQL] {sql}")
    
    # 安全檢查
    upper_sql = sql.upper().strip()
    if not upper_sql.startswith("SELECT") or any(kw in upper_sql for kw in ["DROP", "DELETE", "UPDATE", "INSERT"]):
        print("【安全警報】非安全的查詢語法，拒絕執行！")
        return

    conn = sqlite3.connect(db_name)
    cursor = conn.cursor()
    
    try:
        cursor.execute(sql)
        rows = cursor.fetchall()
        
        if not rows:
            print("查無符合條件的消費紀錄。\n")
            return
            
        columns = [desc[0] for desc in cursor.description]
        print("\n" + "=" * 80)
        print(" | ".join([f"{col:<12}" for col in columns]))#標頭
        print("-" * 80)
        
        for row in rows:
            print(" | ".join([f"{str(val):<12}" for val in row]))#查詢出的條目
            
        print("=" * 80)
        print(f"共查詢到 {len(rows)} 筆資料。\n")
        
    except sqlite3.OperationalError as e:
        print(f"SQL 執行失敗: {e}")
    finally:
        conn.close()

# -----------------------------------------------------------------
# 步驟七：終端機互動式主選單
# -----------------------------------------------------------------

# 定義命令列互動選單函式
def interactive_menu(db_name="expense_test.db"):
    # 使用無窮迴圈讓使用者可以連續操作多個功能
    while True:
        # 印出功能選項清單
        print("====== 智慧記帳分析選單 ======")
        print("1. 查看全部記帳紀錄")
        print("2. 查詢特定日期的花費 (如 2026-11-04)")
        print("3. 查詢特定地點/通路的消費 (如 一般門市、全家便利商店)")
        print("4. 查詢特定對象的消費 (如 自己、媽媽)")
        print("5. 依通路/地點統計總支出")
        print("6. 依對象統計總支出")
        print("7. 手動輸入文字由ai查詢")
        print("8. 麥克風語音搜尋")
        print("0. 離開系統")
        
        # 接收使用者輸入的代號
        choice = input("請選擇功能代號 (0-8): ").strip()
        
        # 依代號執行對應動作
        if choice == "1":
            list_all_expenses(db_name)
        elif choice == "2":
            d = input("請輸入日期 (YYYY-MM-DD): ").strip()
            query_expenses(db_name, date=d)
        elif choice == "3":
            c = input("請輸入通路或店名關鍵字: ").strip()
            query_expenses(db_name, channel=c)
        elif choice == "4":
            p = input("請輸入消費對象 (如 自己 / 媽媽): ").strip()
            query_expenses(db_name, target_person=p)
        elif choice == "5":
            show_summary_by_dimension(db_name, dimension="channel")
        elif choice == "6":
            show_summary_by_dimension(db_name, dimension="target_person")
        elif choice == "7":
            q = input("請輸入查詢內容: ").strip()
            execute_natural_query(q, db_name)
        elif choice == "8":
            spoken_text = record_voice_to_text()
            if spoken_text:
                execute_natural_query(spoken_text, db_name)
        elif choice == "0":
            # 退出系統並跳出迴圈
            print("退出系統。")
            break
        else:
            # 處理無效輸入
            print("輸入錯誤，請重新選擇！\n")


# -----------------------------------------------------------------
# 程式進入點
# -----------------------------------------------------------------

# 判斷是否為直接執行此腳本
if __name__ == "__main__":
    # 初始化或檢查資料庫結構
    init_db()

    # 取得本程式所在資料夾的絕對路徑，拼接測試圖片完整路徑
    current_dir = os.path.dirname(os.path.abspath(__file__))
    test_image_file = os.path.join(current_dir, "test_receipt.jpg")

    # 詢問使用者是否要辨識新圖片
    run_ocr = input("是否要辨識新收據圖片？(y/n，預設 n): ").strip().lower()
    
    # 若使用者選擇 y 且圖片確實存在，執行 AI 辨識並存入資料庫
    if run_ocr == 'y' and os.path.exists(test_image_file):
        result_dict = parse_receipt_with_ai(test_image_file)
        new_id = save_expense_to_db(result_dict)
        print(f"成功存入資料庫！資料 ID 為: {new_id}")

    # 啟動命令列互動選單供使用者查詢與分析
    interactive_menu()