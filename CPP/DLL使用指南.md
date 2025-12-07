# ClassifierDLL 使用指南

C# WPF 调用 LightGBM 分类器 DLL 的完整指南。

---

## 📋 目录

1. [编译 DLL](#编译-dll)
2. [C# 项目配置](#c-项目配置)
3. [基本使用](#基本使用)
4. [API 参考](#api-参考)
5. [修改和重新编译](#修改和重新编译)
6. [常见问题](#常见问题)

---

## 编译 DLL

### 前置要求

- **CMake** 3.15+
- **Visual Studio 2019+**
- **ONNX Runtime** 1.19.2+（必须使用新版本）

### 步骤 1: 安装 ONNX Runtime

1. 下载最新版本：https://github.com/microsoft/onnxruntime/releases
2. 下载 `onnxruntime-win-x64-*.zip`
3. 解压到 `C:\onnxruntime`

### 步骤 2: 配置 CMake

```bash
cd CPP

# 配置项目
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DONNXRUNTIME_DIR=C:/onnxruntime
```

### 步骤 3: 编译 DLL

```bash
# 编译
cmake --build build --config Release
```

**编译时间**：约 10-30 秒

### 步骤 4: 查找生成的文件

编译完成后，在 `build\bin\Release\` 目录下会生成：

```
build/bin/Release/
├── ClassifierDLL.dll          # 主 DLL 文件
├── ClassifierDLL.lib          # 导入库（C++ 项目需要）
├── onnxruntime.dll            # ONNX Runtime 依赖
├── lightgbm_model.onnx        # 模型文件（28.28 MB）
├── scaler_params.json         # 标准化参数
└── label_mapping.json         # 类别映射
```

**重要**: 所有这些文件（除了 .lib）都需要复制到你的 C# 项目的输出目录！

---

## C# 项目配置

### 步骤 1: 创建 WPF 项目

在 Visual Studio 中创建新的 WPF 应用程序项目（.NET 6.0）。

### 步骤 2: 设置平台为 x64

1. 在工具栏选择"配置管理器"
2. 将平台设置为 **x64**（不是 AnyCPU 或 x86）
3. 或者在 `.csproj` 文件中添加：
   ```xml
   <Platforms>x64</Platforms>
   ```

### 步骤 3: 添加 Classifier.cs

将 `ClassifierWrapper.cs` 或 `Classifier.cs` 文件添加到你的项目中。

### 步骤 4: 复制 DLL 和依赖文件

将以下文件复制到你的项目输出目录：

**Debug 模式：**
```bash
copy ..\CPP\build\bin\Release\ClassifierDLL.dll bin\x64\Debug\net6.0-windows\
copy ..\CPP\build\bin\Release\onnxruntime.dll bin\x64\Debug\net6.0-windows\
copy ..\CPP\build\bin\Release\lightgbm_model.onnx bin\x64\Debug\net6.0-windows\
copy ..\CPP\build\bin\Release\scaler_params.json bin\x64\Debug\net6.0-windows\
copy ..\CPP\build\bin\Release\label_mapping.json bin\x64\Debug\net6.0-windows\
```

**Release 模式：**
```bash
copy ..\CPP\build\bin\Release\*.dll bin\x64\Release\net6.0-windows\
copy ..\CPP\build\bin\Release\*.onnx bin\x64\Release\net6.0-windows\
copy ..\CPP\build\bin\Release\*.json bin\x64\Release\net6.0-windows\
```

**提示**: 可以在项目属性中设置这些文件为"复制到输出目录"，这样每次编译都会自动复制。

---

## 基本使用

### 最简单的示例

```csharp
using ClassifierDemo;  // 或你的命名空间

// 1. 创建分类器实例
using (var classifier = new Classifier())
{
    // 2. 初始化
    classifier.InitializeClassifier(
        "lightgbm_model.onnx",
        "scaler_params.json",
        "label_mapping.json"
    );

    // 3. 准备特征数据（20个特征）
    float[] features = new float[20]
    {
        581.722f, -0.411162f, -0.262966f, 0.1029f, 355.387f,
        // ... 其他15个特征
    };

    // 4. 预测
    var result = classifier.Predict(features);

    // 5. 使用结果
    Console.WriteLine($"预测类别: {result.ClassName}");
    Console.WriteLine($"置信度: {result.Confidence:P2}");
}
```

### 批量 CSV 预测示例

```csharp
using (var classifier = new Classifier())
{
    // 初始化
    classifier.InitializeClassifier(
        "lightgbm_model.onnx",
        "scaler_params.json",
        "label_mapping.json"
    );

    // 批量预测 CSV 文件
    var results = classifier.PredictFromCSV(@"D:\data\TEST.csv");

    Console.WriteLine($"总共处理了 {results.SampleCount} 个样本\n");

    // 显示每个样本的结果
    for (int i = 0; i < results.SampleCount; i++)
    {
        var result = results.Results[i];
        Console.WriteLine($"[样本 {i + 1}]");
        Console.WriteLine($"  预测: {result.ClassName}");
        Console.WriteLine($"  置信度: {result.Confidence:P2}");
    }
}
```

---

## API 参考

### Classifier 类

#### InitializeClassifier

```csharp
bool InitializeClassifier(string modelPath, string scalerPath, string labelPath)
```

初始化分类器。必须在预测之前调用。

**参数:**
- `modelPath`: ONNX 模型文件路径
- `scalerPath`: 标准化参数文件路径
- `labelPath`: 类别映射文件路径

**返回:** 成功返回 true

**异常:**
- `Exception`: 初始化失败时抛出，包含详细错误信息

**错误代码:**
- `-1`: 无法打开标准化参数文件
- `-2`: 标准化参数无效（需要20个特征）
- `-3`: 类别映射无效（需要6个类别）
- `-999`: 未知错误

#### Predict

```csharp
PredictionResult Predict(float[] features)
```

预测单个样本。

**参数:**
- `features`: 包含20个特征值的数组

**返回:** `PredictionResult` 对象

**异常:**
- `InvalidOperationException`: 分类器未初始化
- `ArgumentException`: 特征数组长度不是20
- `Exception`: 预测失败

**错误代码:**
- `-1`: 分类器未初始化
- `-2`: 特征数量无效
- `-999`: 预测过程中发生未知错误

#### PredictFromCSV

```csharp
BatchPredictionResult PredictFromCSV(string csvPath)
```

批量预测 CSV 文件。

**参数:**
- `csvPath`: CSV 文件路径

**返回:** `BatchPredictionResult` 对象

**异常:**
- `InvalidOperationException`: 分类器未初始化
- `ArgumentException`: CSV 文件路径为空
- `Exception`: 批量预测失败

**错误代码:**
- `-1`: 分类器未初始化
- `-2`: CSV 文件为空或无有效数据
- `-999`: 预测过程中发生未知错误

**CSV 文件格式要求:**
- 每行 20 个特征值
- 用逗号分隔
- 无表头行
- 纯数字数据

#### Dispose

```csharp
void Dispose()
```

释放资源。建议使用 `using` 语句自动调用。

---

### PredictionResult 类

预测结果对象。

#### 属性

| 属性 | 类型 | 说明 |
|------|------|------|
| `ClassIndex` | `int` | 预测的类别索引 (0-5) |
| `ClassName` | `string` | 预测的类别名称 |
| `Confidence` | `float` | 置信度 (0-1) |
| `Probabilities` | `float[]` | 所有类别的概率分布 (长度为6) |

**类别索引对应关系:**
- 0: DH (电荷)
- 1: KD (空洞)
- 2: PS10 (聚苯乙烯10)
- 3: PS10-H (聚苯乙烯10-H)
- 4: QZ (球状)
- 5: YM (圆盘)

---

### BatchPredictionResult 类

批量预测结果对象。

#### 属性

| 属性 | 类型 | 说明 |
|------|------|------|
| `SampleCount` | `int` | 样本数量 |
| `Results` | `PredictionResult[]` | 每个样本的预测结果数组 |

---

## 修改和重新编译

### 场景1：修改 C++ DLL 代码

如果你修改了 `ClassifierDLL.cpp`：

```bash
# 1. 关闭所有正在运行的 WPF 程序（重要！）

# 2. 进入 CPP 目录
cd CPP

# 3. 重新编译
cmake --build build --config Release

# 4. 复制新的 DLL 到 WPF 项目
copy build\bin\Release\ClassifierDLL.dll ..\WPF_Classifier_Demo\bin\x64\Debug\net6.0-windows\

# 5. 重新运行 WPF 程序
```

**重要提示：**
- ⚠️ 必须先关闭 WPF 程序，否则 DLL 被占用无法覆盖
- ⚠️ 每次修改 C++ 代码都必须重新编译 DLL
- ⚠️ 新的 DLL 必须复制到 WPF 程序的输出目录

### 场景2：修改 C# 代码

如果你修改了 WPF 项目的 C# 代码：

**使用 Visual Studio：**
- 直接按 F5，Visual Studio 会自动重新编译

**使用命令行：**
```bash
cd WPF_Classifier_Demo
dotnet build -c Debug
dotnet run
```

**注意**：修改 C# 代码不需要重新编译 C++ DLL。

### 场景3：修改模型

如果你重新训练了模型：

```bash
# 1. 重新训练和导出
python train_classifier.py
python export_to_onnx.py

# 2. 复制新模型到 C++ 项目
copy lightgbm_model.onnx CPP\
copy scaler_params.json CPP\
copy label_mapping.json CPP\

# 3. 重新编译 C++ 项目（会自动复制模型文件）
cd CPP
cmake --build build --config Release

# 4. 复制到 WPF 项目
copy build\bin\Release\*.onnx ..\WPF_Classifier_Demo\bin\x64\Debug\net6.0-windows\
copy build\bin\Release\*.json ..\WPF_Classifier_Demo\bin\x64\Debug\net6.0-windows\

# 5. 重新运行 WPF 程序测试新模型
```

### 场景4：修改 CMakeLists.txt

如果你修改了 `CMakeLists.txt`（比如添加新的源文件或修改编译选项）：

```bash
# 1. 删除旧的构建目录
cd CPP
rm -rf build

# 2. 重新配置
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DONNXRUNTIME_DIR=C:/onnxruntime

# 3. 编译
cmake --build build --config Release

# 4. 复制 DLL 到 WPF 项目
copy build\bin\Release\ClassifierDLL.dll ..\WPF_Classifier_Demo\bin\x64\Debug\net6.0-windows\
```

### 场景5：更换 ONNX Runtime 版本

如果你想升级 ONNX Runtime：

```bash
# 1. 下载新版本的 ONNX Runtime
# 2. 解压到 C:\onnxruntime（覆盖旧版本）

# 3. 删除旧的构建目录
cd CPP
rm -rf build

# 4. 重新配置和编译
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DONNXRUNTIME_DIR=C:/onnxruntime
cmake --build build --config Release

# 5. 复制新的 DLL 到 WPF 项目
copy build\bin\Release\*.dll ..\WPF_Classifier_Demo\bin\x64\Debug\net6.0-windows\
```

---

## 完整的工作流程

### 从零开始的完整流程

```bash
# === 第一部分：训练模型 ===
# 1. 训练模型
python train_classifier.py

# 2. 导出 ONNX 模型
python export_to_onnx.py

# === 第二部分：编译 C++ DLL ===
# 3. 配置 CMake
cd CPP
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DONNXRUNTIME_DIR=C:/onnxruntime

# 4. 编译 DLL
cmake --build build --config Release

# === 第三部分：运行 WPF 程序 ===
# 5. 编译 WPF 项目
cd ..\WPF_Classifier_Demo
dotnet build -c Debug

# 6. 复制 DLL 和模型文件
copy ..\CPP\build\bin\Release\*.dll bin\x64\Debug\net6.0-windows\
copy ..\CPP\build\bin\Release\*.onnx bin\x64\Debug\net6.0-windows\
copy ..\CPP\build\bin\Release\*.json bin\x64\Debug\net6.0-windows\

# 7. 运行程序
dotnet run
```

---

## 常见问题

### Q1: 运行时提示找不到 ClassifierDLL.dll

**原因：** DLL 不在程序目录或平台不匹配。

**解决方案：**
1. 检查 DLL 是否在 `bin\x64\Debug\net6.0-windows\` 目录
2. 确保编译的是 **x64** 平台（不是 x86 或 AnyCPU）
3. 检查 DLL 是否被杀毒软件拦截

### Q2: 初始化失败，错误代码 -1

**原因：** 无法打开 `scaler_params.json` 文件。

**解决方案：**
1. 检查文件是否存在
2. 检查路径是否正确（可以使用绝对路径）
3. 确保使用最新的 DLL（已修复路径编码问题）

### Q3: 初始化失败，错误代码 -2

**原因：** 标准化参数无效。

**解决方案：**
确保 `scaler_params.json` 包含20个特征的 mean 和 scale。

### Q4: 初始化失败，错误代码 -3

**原因：** 类别映射无效。

**解决方案：**
确保 `label_mapping.json` 包含6个类别。

### Q5: 预测失败，错误代码 -2

**原因：** 特征数量不是20。

**解决方案：**
确保传入的数组长度为20。

### Q6: 批量预测失败，错误代码 -2

**原因：** CSV 文件为空或格式不正确。

**解决方案：**
1. 检查 CSV 文件格式（20个特征，逗号分隔，无表头）
2. 不要使用 Excel 格式的 CSV，必须是纯文本
3. 确保使用最新的 DLL（已修复路径编码问题）

### Q7: DLL 加载很慢（超过 1 分钟）

**原因：** ONNX Runtime 版本过旧。

**解决方案：**
1. 下载最新版本的 ONNX Runtime (1.19.2+)
2. 重新编译 DLL（参考"场景5：更换 ONNX Runtime 版本"）

### Q8: 复制 DLL 时提示"文件正在使用"

**原因：** WPF 程序正在运行，DLL 被占用。

**解决方案：**
1. 关闭 WPF 程序
2. 重新复制 DLL
3. 再次运行 WPF 程序

### Q9: Visual Studio 编译失败

**原因：** 缺少 .NET 6.0 SDK 或平台设置错误。

**解决方案：**
1. 安装 .NET 6.0 SDK
2. 确保平台设置为 **x64**
3. 检查 `.csproj` 文件中是否包含 `<Platforms>x64</Platforms>`

### Q10: 预测结果都是 100% 置信度

**原因：** 这是正常的！

**说明：**
如果测试数据与训练数据中的某个类别非常相似，模型会给出接近 100% 的置信度。这说明：
- ✅ 模型工作正常
- ✅ 测试数据质量好
- ✅ 模型对这些样本很有把握

---

## 性能指标

- **初始化时间**: < 1 秒（使用新版 ONNX Runtime）
- **单次预测**: < 1 毫秒
- **批量预测**: 支持最多 1000 个样本
- **内存占用**: ~30 MB（主要是模型）
- **准确率**: 81.65%

---

## 类别说明

| 索引 | 类别名称 | 说明 |
|------|----------|------|
| 0 | DH | 电荷 |
| 1 | KD | 空洞 |
| 2 | PS10 | 聚苯乙烯10 |
| 3 | PS10-H | 聚苯乙烯10-H |
| 4 | QZ | 球状 |
| 5 | YM | 圆盘 |

---

## 技术细节

### DLL 导出函数

#### Initialize
```cpp
int Initialize(const wchar_t* model_path,
               const wchar_t* scaler_path,
               const wchar_t* label_path);
```

#### Predict
```cpp
int Predict(const float* features,
            int feature_count,
            float* probabilities,
            int* predicted_class);
```

#### PredictFromCSV
```cpp
int PredictFromCSV(const wchar_t* csv_path,
                   int* predicted_classes,
                   float* all_probabilities,
                   int* sample_count);
```

#### GetClassName
```cpp
int GetClassName(int class_index,
                 wchar_t* buffer,
                 int buffer_size);
```

#### Cleanup
```cpp
void Cleanup();
```

### 内存管理

- DLL 使用全局变量存储模型会话
- 调用 `Initialize` 时分配资源
- 调用 `Cleanup` 或 `Dispose` 时释放资源
- 建议使用 C# 的 `using` 语句自动管理资源

### 线程安全

**当前实现不是线程安全的**，因为使用了全局变量。

如需多线程支持：
1. 每个线程创建独立的 Classifier 实例
2. 或者在调用时加锁

---

## 更新日志

### v1.1.0 (2025-12-06)
- ✅ 修复了文件路径编码问题（使用 wifstream）
- ✅ 添加了批量 CSV 预测功能
- ✅ 优化了模型加载速度
- ✅ 完善了错误处理和异常信息

### v1.0.0 (2025-12-06)
- ✅ 实现基本的 DLL 功能
- ✅ 支持单样本预测
- ✅ 提供 C# 包装类

---

## 技术支持

如有问题，请查看：
1. 本文档的"常见问题"部分
2. 主项目的 README.md
3. WPF_Classifier_Demo/README.md

---

## 许可证

本项目遵循 MIT 许可证。
