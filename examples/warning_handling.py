#!/usr/bin/env python3
"""
處理 lmfit 警告的示例

展示如何在 Josephson Junction 擬合中處理和抑制常見的數值計算警告。
"""

import numpy as np
import warnings
from josephson_fit import (
    JosephsonTripleFitter,
    generate_synthetic_data,
    configure_josephson_warnings,
    suppress_warnings,
    get_warning_summary
)
from josephson_fit.models import Model2


def example_with_warnings():
    """展示如何處理警告的示例"""
    
    print("🔧 Josephson Junction 擬合警告處理示例")
    print("=" * 50)
    
    # 生成測試數據
    phi_ext = np.linspace(-2, 2, 150)
    params = {
        'Ic': 2.0e-6, 'T': 0.4, 'f': 1.2, 'd': 0.1, 'phi0': 0.3,
        'k': 0.01, 'r': -0.005, 'C': 0.05
    }
    
    current, _ = generate_synthetic_data(
        phi_ext, Model2().function, params, noise_level=0.02
    )
    
    print("\n1. 顯示警告摘要")
    print("-" * 30)
    summary = get_warning_summary()
    
    print("📋 常見警告類型:")
    for warning_type, info in summary.items():
        if isinstance(info, dict):
            print(f"\n• {warning_type}:")
            for key, value in info.items():
                print(f"  - {key}: {value}")
    
    print("\n💡 一般建議:")
    for advice in summary["general_advice"]:
        print(f"  • {advice}")
    
    print("\n2. 方法一：全局抑制警告")
    print("-" * 30)
    configure_josephson_warnings(verbose=False)
    
    fitter = JosephsonTripleFitter(phi_ext, current)
    results = fitter.fit_all_models()
    
    print("✅ 擬合完成（警告已抑制）")
    print(f"📊 模型 1 AIC: {results['model1'].aic:.2f}")
    print(f"📊 模型 2 AIC: {results['model2'].aic:.2f}")
    print(f"📊 模型 3 AIC: {results['model3'].aic:.2f}")
    
    print("\n3. 方法二：上下文管理器抑制警告")
    print("-" * 30)
    
    # 重新啟用警告以演示
    configure_josephson_warnings(verbose=True)
    
    # 使用上下文管理器
    with suppress_warnings():
        fitter2 = JosephsonTripleFitter(phi_ext, current)
        comparison = fitter2.compare_models()
    
    print("✅ 擬合完成（使用上下文管理器抑制警告）")
    best_model = comparison['model_selection']['recommended_model']
    confidence = comparison['model_selection']['confidence']
    print(f"🏆 推薦模型: {best_model}")
    print(f"🎯 信心水平: {confidence}")
    
    print("\n4. 方法三：檢查特定警告原因")
    print("-" * 30)
    
    # 檢查擬合品質來理解警告
    best_result = comparison['model_summary'][best_model]
    
    print("📈 擬合品質指標:")
    print(f"  • R²: {best_result['r_squared']:.4f}")
    print(f"  • 簡化 χ²: {best_result['reduced_chi_square']:.4f}")
    print(f"  • 品質評級: {best_result['quality_rating']}")
    
    # 如果品質評級較低，說明警告可能有意義
    if best_result['quality_rating'] in ['Poor', 'Fair']:
        print("\n⚠️  警告說明:")
        print("  擬合品質較低，信賴區間警告可能反映真實的不確定性問題")
        print("  建議：檢查數據品質、增加數據點或簡化模型")
    else:
        print("\n✅ 擬合品質良好:")
        print("  信賴區間警告通常是數值計算問題，不影響主要結果")
    
    print("\n5. 方法四：手動處理信賴區間")
    print("-" * 30)
    
    # 手動計算更穩健的參數不確定性
    best_fit_result = results[best_model]
    
    print("📊 參數估計 (±1σ 不確定性):")
    for param_name, value in best_fit_result.params.items():
        error = best_fit_result.param_errors.get(param_name)
        if error is not None:
            if param_name == 'Ic':
                print(f"  • {param_name}: {value*1e6:.2f} ± {error*1e6:.2f} µA")
            elif param_name in ['T', 'f']:
                print(f"  • {param_name}: {value:.4f} ± {error:.4f}")
            else:
                print(f"  • {param_name}: {value:.3f} ± {error:.3f}")
        else:
            print(f"  • {param_name}: {value:.4f} (不確定性計算失敗)")
    
    print("\n6. 最佳實踐建議")
    print("-" * 30)
    print("📝 處理警告的建議:")
    print("  1. 對於生產代碼，使用 configure_josephson_warnings(verbose=False)")
    print("  2. 對於調試，使用 configure_josephson_warnings(verbose=True)")
    print("  3. 關注擬合品質指標而非警告數量")
    print("  4. 如果經常出現警告，檢查:")
    print("     • 數據品質和數量")
    print("     • 模型複雜度是否合適")
    print("     • 初始參數猜測是否合理")
    print("     • 參數邊界是否適當")


def example_custom_warning_filter():
    """展示自定義警告過濾的高級用法"""
    
    print("\n🔧 自定義警告過濾示例")
    print("=" * 40)
    
    # 生成有挑戰性的數據（更多噪聲）
    phi_ext = np.linspace(-1, 1, 80)  # 較少的數據點
    params = {'Ic': 1e-6, 'T': 0.8, 'f': 2.5, 'd': 0, 'phi0': 0,
              'k': 0, 'r': 0, 'C': 0}
    
    current, _ = generate_synthetic_data(
        phi_ext, Model2().function, params, noise_level=0.1  # 高噪聲
    )
    
    # 自定義警告處理
    with warnings.catch_warnings(record=True) as warning_list:
        warnings.simplefilter("always")  # 捕獲所有警告
        
        fitter = JosephsonTripleFitter(phi_ext, current)
        result = fitter.fit_model('model2')
        
        # 分析捕獲的警告
        lmfit_warnings = [w for w in warning_list if 'lmfit' in str(w.filename)]
        confidence_warnings = [w for w in lmfit_warnings if 'confidence' in str(w.message)]
        
        print(f"📊 捕獲的警告統計:")
        print(f"  • 總警告數: {len(warning_list)}")
        print(f"  • lmfit 警告: {len(lmfit_warnings)}")
        print(f"  • 信賴區間警告: {len(confidence_warnings)}")
        
        if confidence_warnings:
            print(f"\n⚠️  信賴區間警告詳情:")
            for i, w in enumerate(confidence_warnings[:3]):  # 只顯示前3個
                print(f"  {i+1}. {w.message}")
            
            if len(confidence_warnings) > 3:
                print(f"  ... 還有 {len(confidence_warnings)-3} 個類似警告")
    
    # 評估是否需要關注這些警告
    print(f"\n📈 擬合結果評估:")
    print(f"  • R²: {result.r_squared:.4f}")
    print(f"  • 簡化 χ²: {result.reduced_chi_square:.4f}")
    
    if result.reduced_chi_square > 5 or result.r_squared < 0.8:
        print("  ⚠️  擬合品質較低 - 警告可能指示真實問題")
        print("  💡 建議：增加數據點、降低噪聲或簡化模型")
    else:
        print("  ✅ 擬合品質可接受 - 警告主要是數值計算問題")


if __name__ == "__main__":
    try:
        example_with_warnings()
        example_custom_warning_filter()
        
        print("\n🎉 警告處理示例完成！")
        print("\n📚 更多信息:")
        print("  • 使用 help(configure_josephson_warnings) 查看詳細文檔")
        print("  • 使用 get_warning_summary() 獲取警告說明")
        print("  • 在生產環境中建議抑制數值計算警告")
        
    except Exception as e:
        print(f"❌ 示例執行失敗: {e}")
        print("請確保 josephson_fit 套件正確安裝")
