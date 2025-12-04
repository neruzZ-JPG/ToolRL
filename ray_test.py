import ray
import os

# 正确设置环境变量的方法
# 注意：必须在 ray.init() 之前设置
os.environ["RAY_ADDRESS"] = "auto"

# 调用 init 时不传参数，它会自动读取环境变量 RAY_ADDRESS
ray.init()

print(ray.is_initialized())