"""
部署前文件驗證腳本
運行此腳本檢查所有必要文件是否正確
"""

import os
import sys
from pathlib import Path

def check_file(filename, required_content=None, no_extension=False):
    """檢查文件是否存在且內容正確"""
    print(f"\n檢查 {filename}...")
    
    # 檢查文件存在
    if not os.path.exists(filename):
        print(f"  ❌ 文件不存在: {filename}")
        return False
    
    # 檢查副檔名
    if no_extension:
        if '.' in filename:
            print(f"  ⚠️  警告: {filename} 不應該有副檔名!")
            print(f"     當前: {filename}")
            print(f"     應該是: {filename.split('.')[0]}")
            return False
    
    # 讀取內容
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            content = f.read()
    except Exception as e:
        print(f"  ❌ 無法讀取文件: {e}")
        return False
    
    # 檢查內容
    if required_content:
        if isinstance(required_content, list):
            missing = [req for req in required_content if req not in content]
            if missing:
                print(f"  ❌ 缺少必要內容:")
                for item in missing:
                    print(f"     - {item}")
                return False
        elif required_content not in content:
            print(f"  ❌ 缺少必要內容: {required_content}")
            return False
    
    # 顯示文件信息
    lines = content.split('\n')
    size = os.path.getsize(filename)
    print(f"  ✓ 文件存在")
    print(f"    行數: {len(lines)}")
    print(f"    大小: {size} bytes")
    
    # 顯示前幾行
    print(f"    內容預覽:")
    for i, line in enumerate(lines[:3], 1):
        if line.strip():
            print(f"      {i}. {line[:60]}{'...' if len(line) > 60 else ''}")
    
    return True

def check_python_files():
    """檢查 Python 文件"""
    print("\n" + "="*70)
    print("檢查 Python 代碼文件")
    print("="*70)
    
    files = {
        'btc_trading_system.py': ['class NewsCollector', 'class BTCPredictorWithPersistence'],
        'btc_api_server.py': ['FastAPI', 'app = FastAPI', '@app.get']
    }
    
    all_good = True
    for filename, required in files.items():
        if not check_file(filename, required):
            all_good = False
    
    return all_good

def check_deployment_files():
    """檢查部署文件"""
    print("\n" + "="*70)
    print("檢查部署配置文件")
    print("="*70)
    
    checks = {
        'requirements.txt': {
            'required': ['fastapi', 'uvicorn', 'pandas', 'scikit-learn'],
            'no_extension': False
        },
        'Procfile': {
            'required': ['web:', 'uvicorn', 'btc_api_server:app'],
            'no_extension': True
        },
        'runtime.txt': {
            'required': ['python-3'],
            'no_extension': False
        }
    }
    
    all_good = True
    for filename, config in checks.items():
        if not check_file(
            filename, 
            config['required'], 
            config['no_extension']
        ):
            all_good = False
            
            # 提供創建建議
            print(f"\n  💡 如何創建 {filename}:")
            if filename == 'Procfile':
                print(f"     1. 用記事本創建新文件")
                print(f"     2. 輸入: web: uvicorn btc_api_server:app --host 0.0.0.0 --port $PORT")
                print(f"     3. 另存為 'Procfile' (無副檔名!)")
                print(f"     4. 存檔類型選「所有檔案」")
            elif filename == 'requirements.txt':
                print(f"     使用命令: python create_requirements.py")
            elif filename == 'runtime.txt':
                print(f"     使用命令: echo python-3.10.12 > runtime.txt")
    
    return all_good

def check_gitignore():
    """檢查 .gitignore"""
    print("\n" + "="*70)
    print("檢查 .gitignore")
    print("="*70)
    
    required = ['__pycache__', '*.pkl', '.vscode']
    result = check_file('.gitignore', required)
    
    if not result:
        print("\n  💡 創建 .gitignore:")
        print("     Windows: New-Item -Path . -Name '.gitignore' -ItemType 'file'")
        print("     Linux/Mac: touch .gitignore")
    
    return result

def check_structure():
    """檢查項目結構"""
    print("\n" + "="*70)
    print("檢查項目結構")
    print("="*70)
    
    required_files = [
        'btc_trading_system.py',
        'btc_api_server.py',
        'requirements.txt',
        'Procfile',
        'runtime.txt'
    ]
    
    optional_files = [
        '.gitignore',
        'README.md'
    ]
    
    print("\n必需文件:")
    all_exist = True
    for f in required_files:
        exists = os.path.exists(f)
        print(f"  {'✓' if exists else '❌'} {f}")
        if not exists:
            all_exist = False
    
    print("\n可選文件:")
    for f in optional_files:
        exists = os.path.exists(f)
        print(f"  {'✓' if exists else '○'} {f}")
    
    # 檢查 models 資料夾
    models_dir = Path('models')
    if models_dir.exists():
        print(f"\n✓ models/ 資料夾存在")
        pkl_files = list(models_dir.glob('*.pkl'))
        if pkl_files:
            print(f"  ⚠️  警告: 發現 {len(pkl_files)} 個 .pkl 文件 (會被 .gitignore)")
    else:
        print(f"\n○ models/ 資料夾不存在 (部署後會自動創建)")
    
    return all_exist

def provide_next_steps():
    """提供下一步操作"""
    print("\n" + "="*70)
    print("下一步操作")
    print("="*70)
    
    print("""
1. 如果所有檢查都通過 ✓，執行:
   
   git init
   git add .
   git commit -m "Initial commit"
   
2. 創建 GitHub 倉庫並推送:
   
   git remote add origin https://github.com/你的用戶名/btc-trading-api.git
   git push -u origin main
   
3. 在 Railway 部署:
   
   方法 A: 訪問 railway.app → Deploy from GitHub
   方法 B: railway init && railway up
   
4. 獲取 URL 並更新 iOS App:
   
   在 Swift 代碼中更新 baseURL
    """)

def main():
    """主函數"""
    print("="*70)
    print("🔍 Railway 部署文件驗證工具")
    print("="*70)
    print(f"當前目錄: {os.getcwd()}")
    
    checks = [
        ("項目結構", check_structure),
        ("Python 文件", check_python_files),
        ("部署文件", check_deployment_files),
        (".gitignore", check_gitignore)
    ]
    
    results = {}
    for name, check_func in checks:
        try:
            results[name] = check_func()
        except Exception as e:
            print(f"\n❌ 檢查 {name} 時出錯: {e}")
            results[name] = False
    
    # 總結
    print("\n" + "="*70)
    print("檢查總結")
    print("="*70)
    
    for name, passed in results.items():
        status = "✓ 通過" if passed else "❌ 失敗"
        print(f"{status} - {name}")
    
    all_passed = all(results.values())
    
    if all_passed:
        print("\n🎉 所有檢查通過！可以開始部署了！")
        provide_next_steps()
        return 0
    else:
        print("\n⚠️  有些檢查未通過，請修復後重新運行此腳本")
        print("\n修復建議:")
        print("1. 確保所有文件都已創建")
        print("2. Procfile 不能有副檔名")
        print("3. 文件內容格式正確")
        print("4. 編碼使用 UTF-8")
        return 1

if __name__ == "__main__":
    sys.exit(main())
