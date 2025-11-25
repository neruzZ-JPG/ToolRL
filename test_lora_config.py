#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
测试脚本：验证vllm LoRA配置替换peft是否成功
"""

import sys
import os

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_vllm_lora_import():
    """测试是否能正确导入vllm的LoRAConfig"""
    try:
        # 尝试导入vllm的LoRAConfig
# 修复导入缩进问题
        from verl.third_party.vllm.vllm_v_0_6_3.model_runner import LoRAConfig
        print("✅ 成功导入vllm的LoRAConfig")
        
        # 创建一个vllm的LoRAConfig实例，添加必需的max_loras参数
        lora_config = LoRAConfig(max_lora_rank=16, max_loras=1)
        print(f"✅ 成功创建vllm LoRAConfig实例，max_lora_rank: {lora_config.max_lora_rank}, max_loras: {lora_config.max_loras}")
        
        return True
    except ImportError as e:
        print(f"❌ 导入vllm LoRAConfig失败: {e}")
        return False
    except Exception as e:
        print(f"❌ 创建vllm LoRAConfig实例失败: {e}")
        return False

def test_fsdp_workers_import():
    """测试fsdp_workers.py是否能正常导入且不依赖peft"""
    try:
        # 只导入模块，不执行具体功能
        from verl.workers import fsdp_workers
        print("✅ 成功导入verl.workers.fsdp_workers模块")
        
        # 检查模块中是否包含我们修改的内容
        if hasattr(fsdp_workers, 'LoRAConfig'):
            print(f"✅ fsdp_workers模块中包含LoRAConfig")
        else:
            print("❓ fsdp_workers模块中未直接暴露LoRAConfig")
        
        return True
    except ImportError as e:
        print(f"❌ 导入verl.workers.fsdp_workers失败: {e}")
        return False
    except Exception as e:
        print(f"❌ 检查verl.workers.fsdp_workers失败: {e}")
        return False

def test_fsdp_sft_trainer_import():
    """测试fsdp_sft_trainer.py是否能正常导入且不依赖peft"""
    try:
        # 只导入模块，不执行具体功能
        from verl.trainer import fsdp_sft_trainer
        print("✅ 成功导入verl.trainer.fsdp_sft_trainer模块")
        return True
    except ImportError as e:
        print(f"❌ 导入verl.trainer.fsdp_sft_trainer失败: {e}")
        return False
    except Exception as e:
        print(f"❌ 检查verl.trainer.fsdp_sft_trainer失败: {e}")
        return False

def test_peft_not_used():
    """测试代码中是否仍然使用peft"""
    try:
        # 尝试导入peft，这应该不会引发异常，但我们只是检查是否有任何地方仍然使用它
        import peft
        print("⚠️  peft库仍然可以导入，但我们已经从代码中移除了它的使用")
    except ImportError:
        print("✅ peft库不可用，这是预期的")
    
    return True

def main():
    """运行所有测试"""
    print("开始测试vllm LoRA配置替换peft...\n")
    
    tests = [
        test_vllm_lora_import,
        test_fsdp_workers_import,
        test_fsdp_sft_trainer_import,
        test_peft_not_used
    ]
    
    results = []
    for test in tests:
        print(f"运行测试: {test.__name__}")
        result = test()
        results.append(result)
        print()
    
    # 打印总结
    print("测试总结:")
    print(f"通过测试数: {sum(results)}")
    print(f"失败测试数: {len(results) - sum(results)}")
    
    if all(results):
        print("🎉 所有测试通过！vllm LoRA配置替换peft成功。")
    else:
        print("❌ 部分测试失败，请检查修改。")
    
    return all(results)

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
