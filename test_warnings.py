#!/usr/bin/env python3
"""
測試 lmfit 警告處理是否有效
"""

import numpy as np
import warnings
from josephson_fit import (
    JosephsonTripleFitter,
    generate_synthetic_data,
    configure_josephson_warnings,
    suppress_warnings
)
from josephson_fit.models import Model2


def test_warning_suppression():
    """測試警告抑制是否正常工作"""
    
    print("🧪 測試 lmfit 警告處理")
    print("=" * 30)
    
    # 生成測試數據
    phi_ext = np.linspace(-2, 2, 100)
    params = {
        'Ic': 1.5e-6, 'T': 0.6, 'f': 1.3, 'd': 0.05, 'phi0': 0.1,
        'k': 0.02, 'r': -0.01, 'C': 0.08
    }
    
    current, _ = generate_synthetic_data(
        phi_ext, Model2().function, params, noise_level=0.03
    )
    
    print("1. 測試默認警告抑制")
    print("-" * 25)
    
    # 默認情況下應該已經抑制了警告
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        
        fitter = JosephsonTripleFitter(phi_ext, current)
        results = fitter.fit_all_models()
        
        # 計算 lmfit 相關警告
        lmfit_warnings = [warning for warning in w 
                         if 'lmfit' in str(warning.filename).lower() 
                         or 'confidence' in str(warning.message).lower()]
        
        print(f"📊 總警告數: {len(w)}")
        print(f"📊 lmfit 相關警告: {len(lmfit_warnings)}")
        
        if len(lmfit_warnings) == 0:
            print("✅ 警告成功抑制！")
        else:
            print("⚠️  仍有警告，顯示前3個:")
            for i, warning in enumerate(lmfit_warnings[:3]):
                print(f"  {i+1}. {warning.message}")
    
    print("\n2. 測試手動啟用警告")
    print("-" * 25)
    
    # 手動啟用警告來比較
    configure_josephson_warnings(verbose=True)
    
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        
        fitter2 = JosephsonTripleFitter(phi_ext, current)
        result = fitter2.fit_model('model2')
        
        lmfit_warnings = [warning for warning in w 
                         if 'lmfit' in str(warning.filename).lower() 
                         or 'confidence' in str(warning.message).lower()]
        
        print(f"📊 啟用警告後總數: {len(w)}")
        print(f"📊 lmfit 相關警告: {len(lmfit_warnings)}")
        
        if len(lmfit_warnings) > 0:
            print("✅ 警告正常顯示（驗證抑制功能）")
        else:
            print("ℹ️  當前數據沒有觸發警告")
    
    print("\n3. 測試上下文管理器")
    print("-" * 25)
    
    with suppress_warnings():
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            
            fitter3 = JosephsonTripleFitter(phi_ext, current)
            result = fitter3.fit_model('model3')
            
            lmfit_warnings = [warning for warning in w 
                             if 'lmfit' in str(warning.filename).lower() 
                             or 'confidence' in str(warning.message).lower()]
            
            print(f"📊 上下文管理器中警告數: {len(lmfit_warnings)}")
            
            if len(lmfit_warnings) == 0:
                print("✅ 上下文管理器成功抑制警告！")
            else:
                print("⚠️  上下文管理器未完全抑制警告")
    
    print("\n4. 檢查擬合結果品質")
    print("-" * 25)
    
    # 檢查擬合結果以確保功能正常
    best_result = min(results.values(), key=lambda x: x.aic if x else float('inf'))
    
    if best_result:
        print(f"✅ 擬合成功完成")
        print(f"📊 最佳模型 AIC: {best_result.aic:.2f}")
        print(f"📊 R²: {best_result.r_squared:.4f}")
        print(f"📊 簡化 χ²: {best_result.reduced_chi_square:.4f}")
        
        # 檢查關鍵參數
        if 'Ic' in best_result.params:
            Ic_fit = best_result.params['Ic']
            Ic_true = params['Ic']
            relative_error = abs(Ic_fit - Ic_true) / Ic_true
            print(f"📊 Ic 相對誤差: {relative_error*100:.1f}%")
            
            if relative_error < 0.2:  # 20% 容差
                print("✅ 參數擬合精度良好")
            else:
                print("⚠️  參數擬合精度較低，可能需要更多數據")
    else:
        print("❌ 擬合失敗")
    
    # 恢復默認設置
    configure_josephson_warnings(verbose=False)
    print("\n✅ 測試完成，已恢復默認警告設置")


if __name__ == "__main__":
    test_warning_suppression()
