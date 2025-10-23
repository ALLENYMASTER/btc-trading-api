"""
自動創建所有部署需要的文件
運行此腳本將自動生成 requirements.txt, Procfile, runtime.txt, .gitignore
"""

import os
from pathlib import Path

def create_requirements():
    """創建 requirements.txt"""
    content = """fastapi==0.104.1
uvicorn[standard]==0.24.0
pandas==2.1.3
numpy==1.24.3
scikit-learn==1.3.2
textblob==0.17.1
requests==2.31.0
pydantic==2.5.0
python-multipart==0.0.6
"""
    
    with open('requirements.txt', 'w', encoding='utf-8') as f:
        f.write(content)
    
    print("✓ 創建 requirements.txt")

def create_procfile():
    """創建 Procfile (無副檔名)"""
    content = "web: uvicorn btc_api_server:app --host 0.0.0.0 --port $PORT\n"
    
    # 確保沒有副檔名
    with open('Procfile', 'w', encoding='utf-8') as f:
        f.write(content)
    
    print("✓ 創建 Procfile (無副檔名)")

def create_runtime():
    """創建 runtime.txt"""
    content = "python-3.10.12\n"
    
    with open('runtime.txt', 'w', encoding='utf-8') as f:
        f.write(content)
    
    print("✓ 創建 runtime.txt")

def create_gitignore():
    """創建 .gitignore"""
    content = """# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
venv/
env/
ENV/

# Models (太大，不要推送)
models/*.pkl
models/*.json

# IDE
.vscode/
.idea/
*.swp
*.swo

# OS
.DS_Store
Thumbs.db

# Logs
*.log

# Testing
.pytest_cache/
.coverage
"""
    
    with open('.gitignore', 'w', encoding='utf-8') as f:
        f.write(content)
    
    print("✓ 創建 .gitignore")

def create_readme():
    """創建 README.md"""
    content = """# Bitcoin Trading System API

FastAPI backend for BTC trading signal system.

## Features
- Real-time Bitcoin price tracking
- ML-powered trading signals
- Advanced NLP sentiment analysis
- Stop-loss/Take-profit risk management
- Backtesting capabilities

## Deployment
Deployed on Railway: [Your URL]

## API Documentation
Visit `/docs` for interactive API documentation.

## Endpoints
- `GET /price` - Current BTC price
- `GET /signal` - Trading signal
- `GET /news` - Latest analyzed news
- `POST /backtest` - Run backtest simulation
- `POST /model/train` - Retrain model

## Tech Stack
- FastAPI
- scikit-learn
- pandas
- TextBlob
"""
    
    with open('README.md', 'w', encoding='utf-8') as f:
        f.write(content)
    
    print("✓ 創建 README.md")

def create_models_dir():
    """創建 models 資料夾"""
    models_dir = Path('models')
    models_dir.mkdir(exist_ok=True)
    
    # 創建 .gitkeep 保持資料夾在 git 中
    gitkeep = models_dir / '.gitkeep'
    gitkeep.touch()
    
    print("✓ 創建 models/ 資料夾")

def update_api_server():
    """檢查並建議更新 btc_api_server.py"""
    if not os.path.exists('btc_api_server.py'):
        print("⚠️  找不到 btc_api_server.py")
        return
    
    with open('btc_api_server.py', 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 檢查是否需要更新
    needs_update = False
    
    if 'os.environ.get("PORT"' not in content:
        print("\n⚠️  需要更新 btc_api_server.py:")
        print("   在 if __name__ == \"__main__\": 區塊中添加:")
        print("   port = int(os.environ.get('PORT', 8000))")
        needs_update = True
    
    if not needs_update:
        print("✓ btc_api_server.py 已正確配置")

def create_env_example():
    """創建 .env.example"""
    content = """# 環境變量範例
# 複製此文件為 .env 並填入實際值

# Railway 會自動設置 PORT
# PORT=8000

# Python 版本
PYTHON_VERSION=3.10.12

# 模型設置
MODEL_DIR=models
MAX_MODEL_AGE_DAYS=7
"""
    
    with open('.env.example', 'w', encoding='utf-8') as f:
        f.write(content)
    
    print("✓ 創建 .env.example")

def create_deployment_checklist():
    """創建部署檢查清單"""
    content = """# 部署檢查清單

## 部署前
- [ ] 所有 Python 文件已測試
- [ ] requirements.txt 已創建
- [ ] Procfile 已創建 (無副檔名!)
- [ ] runtime.txt 已創建
- [ ] .gitignore 已創建
- [ ] README.md 已更新

## Git 設置
- [ ] git init
- [ ] git add .
- [ ] git commit -m "Initial commit"
- [ ] 創建 GitHub 倉庫
- [ ] git remote add origin [URL]
- [ ] git push -u origin main

## Railway 部署
- [ ] 訪問 railway.app
- [ ] 使用 GitHub 登入
- [ ] Deploy from GitHub repo
- [ ] 選擇倉庫
- [ ] 等待部署完成
- [ ] 生成公開域名
- [ ] 測試 /docs 端點

## 驗證
- [ ] GET /price 返回數據
- [ ] GET /signal 返回信號
- [ ] GET /news 返回新聞
- [ ] API 文檔正常顯示
- [ ] 無 500 錯誤

## iOS App 更新
- [ ] 更新 baseURL 為 Railway URL
- [ ] 測試 App 連接
- [ ] 驗證所有功能

## 完成！🎉
"""
    
    with open('DEPLOYMENT_CHECKLIST.md', 'w', encoding='utf-8') as f:
        f.write(content)
    
    print("✓ 創建 DEPLOYMENT_CHECKLIST.md")

def verify_existing_files():
    """驗證現有文件"""
    print("\n檢查現有文件:")
    print("-" * 50)
    
    required = ['btc_trading_system.py', 'btc_api_server.py']
    missing = []
    
    for f in required:
        if os.path.exists(f):
            print(f"  ✓ {f}")
        else:
            print(f"  ❌ {f} (未找到)")
            missing.append(f)
    
    if missing:
        print(f"\n⚠️  缺少必要文件: {', '.join(missing)}")
        print("   請確保這些文件在當前目錄中")
        return False
    
    return True

def show_next_steps():
    """顯示下一步操作"""
    print("\n" + "="*70)
    print("🎉 設置完成！下一步操作:")
    print("="*70)
    
    print("""
📋 步驟 1: 驗證文件
   運行驗證腳本:
   python verify_deployment_files.py

📋 步驟 2: 初始化 Git
   git init
   git add .
   git commit -m "Initial commit for Railway deployment"

📋 步驟 3: 創建 GitHub 倉庫
   1. 訪問 github.com
   2. 點擊 New repository
   3. 倉庫名: btc-trading-api
   4. 創建後複製 URL

📋 步驟 4: 推送到 GitHub
   git remote add origin https://github.com/你的用戶名/btc-trading-api.git
   git branch -M main
   git push -u origin main

📋 步驟 5: 部署到 Railway
   
   方法 A - 網頁界面 (推薦):
   1. 訪問 railway.app
   2. 用 GitHub 登入
   3. New Project → Deploy from GitHub
   4. 選擇 btc-trading-api
   5. 等待部署完成
   6. Settings → Generate Domain
   7. 複製 URL: https://your-app.railway.app
   
   方法 B - CLI:
   npm install -g @railway/cli
   railway login
   railway init
   railway up
   railway domain

📋 步驟 6: 測試 API
   瀏覽器訪問:
   https://your-app.railway.app/docs
   
   或使用 curl:
   curl https://your-app.railway.app/price

📋 步驟 7: 更新 iOS App
   在 Swift 代碼中修改:
   private let baseURL = "https://your-app.railway.app"

📋 參考文件:
   - DEPLOYMENT_CHECKLIST.md (詳細檢查清單)
   - README.md (項目說明)
   - .env.example (環境變量範例)

💡 提示:
   - Procfile 必須無副檔名
   - 首次部署需要 2-3 分鐘訓練模型
   - 查看日誌: railway logs
   - 遇到問題檢查 Railway Dashboard 的 Logs 標籤
""")

def main():
    """主函數"""
    print("="*70)
    print("🚀 Railway 部署自動設置工具")
    print("="*70)
    print(f"當前目錄: {os.getcwd()}\n")
    
    # 驗證現有文件
    if not verify_existing_files():
        print("\n❌ 請確保 btc_trading_system.py 和 btc_api_server.py 在當前目錄")
        return 1
    
    print("\n開始創建部署文件...")
    print("-" * 50)
    
    try:
        # 創建所有必要文件
        create_requirements()
        create_procfile()
        create_runtime()
        create_gitignore()
        create_readme()
        create_models_dir()
        create_env_example()
        create_deployment_checklist()
        
        print("\n檢查現有配置...")
        print("-" * 50)
        update_api_server()
        
        # 顯示創建的文件
        print("\n創建的文件列表:")
        print("-" * 50)
        files = [
            'requirements.txt',
            'Procfile',
            'runtime.txt',
            '.gitignore',
            'README.md',
            '.env.example',
            'DEPLOYMENT_CHECKLIST.md',
            'models/.gitkeep'
        ]
        
        for f in files:
            if os.path.exists(f):
                size = os.path.getsize(f)
                print(f"  ✓ {f} ({size} bytes)")
        
        # 顯示下一步
        show_next_steps()
        
        return 0
        
    except Exception as e:
        print(f"\n❌ 錯誤: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    import sys
    sys.exit(main())
