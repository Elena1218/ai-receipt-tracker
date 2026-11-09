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

#從 ollama 模組中匯入 Client（客戶端）類別
from ollama import Client

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
    
    # 執行 SQL 指令：如果 expenses 資料表不存在，就建立這張表
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
        # 若找不到檔案，主動拋出 FileNotFoundError 錯誤並停止執行
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
        # 指定要運行的模型名稱
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

    # 清理模型可能多加的 markdown 標記 (例如 ```json 或 ```)
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

    # 嘗試將模型產生的文字解析為 Python 字典
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

    # 回傳結構化後的明細字典
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
        data.get("transaction_date", "2026-11-05"),          # 交易日期
        data.get("category", "未分類"),                       # 分類
        data.get("item_name", "未知品項"),                    # 品項名稱
        float(data.get("amount", 0.0)),                      # 金額 (確保轉為浮點數)
        data.get("target_person", "自己"),                    # 買給誰
        data.get("channel", "一般門市"),                      # 通路
        data.get("raw_ai_response", ""),                     # 原始輸出
        now_time                                             # 建立時間
    )

    # 執行 SQL 插入指令並帶入資料值
    cursor.execute(sql, values)
    
    # 提交資料庫變更
    conn.commit()
    
    # 關閉連線
    conn.close()
    
    # 回傳這筆新資料的 ID
    return record_id


# -----------------------------------------------------------------
# 步驟四：查詢並印出目前資料庫內的所有記帳紀錄
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
    
    # 印出表格表頭
    print("\n" + "=" * 80)
    print(f"{'日期':<12} | {'分類':<8} | {'通路':<10} | {'對象':<6} | {'金額':<8} | {'品項名稱'}")
    print("-" * 80)
    
    # 逐筆遍歷查詢出來的每一列資料
    for row in rows:
        # 解包各欄位變數 欄位取名 方便後續呼叫
        record_id, t_date, cat, item, amt, person, ch = row
        # 格式化輸出各欄位數值
        print(f"{t_date:<12} | {cat:<8} | {ch:<10} | {person:<6} | {amt:<8.1f} | {item}")
        
    print("=" * 80 + "\n")
    
    # 關閉連線
    conn.close()


# -----------------------------------------------------------------
# 程式進入點 (執行測試)
# -----------------------------------------------------------------

# 判斷是否由終端機直接執行本檔案
if __name__ == "__main__":
    # 步驟 1：建立或檢查本地資料庫與資料表
    init_db()

    # 設定用來測試的圖片檔案路徑
    test_image_file = "test_receipt.jpg"

    # 檢查測試圖片是否存在，若存在則執行 AI 辨識流程
    if os.path.exists(test_image_file):
        # 步驟 2：呼叫本地 AI 模型辨識圖片
        result_dict = parse_receipt_with_ai(test_image_file)
        
        # 步驟 3：將辨識出來的結構化資料寫入 SQLite
        new_id = save_expense_to_db(result_dict)
        print(f"成功存入資料庫！資料 ID 為: {new_id}")
    else:
        # 若還沒放圖片，提供提示並改用一組模擬資料寫入，確保能驗證資料庫功能
        print(f"提示：在程式同目錄下找不到 '{test_image_file}'，改以一組模擬資料寫入資料庫測試。")
        mock_data = {
            "transaction_date": "2026-11-05",
            "category": "餐飲",
            "item_name": "無糖黑豆漿、燕麥麵包",
            "amount": 75.0,
            "target_person": "自己",
            "channel": "全家便利商店",
            "raw_ai_response": '{"mock": true}'
        }
        # 寫入模擬資料
        new_id = save_expense_to_db(mock_data)
        print(f"模擬資料已成功寫入資料庫！ID 為: {new_id}")

    # 步驟 4：查看目前資料庫內的記帳明細列表
    list_all_expenses()