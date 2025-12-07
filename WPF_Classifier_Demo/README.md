# WPF 分类器 Demo 项目

简单的 WPF 应用程序，演示如何调用 ClassifierDLL.dll 进行光学特征分类。

---

## 🚀 快速开始

### 方法1：使用 Visual Studio（推荐）

1. **打开项目**：用 Visual Studio 2022 打开 `ClassifierDemo.csproj`
2. **设置平台**：在工具栏选择 **x64** 平台
3. **编译项目**：点击"生成" → "生成解决方案"
4. **复制文件**：
   ```bash
   copy ..\CPP\build\bin\Release\*.dll bin\x64\Debug\net6.0-windows\
   copy ..\CPP\build\bin\Release\*.onnx bin\x64\Debug\net6.0-windows\
   copy ..\CPP\build\bin\Release\*.json bin\x64\Debug\net6.0-windows\
   ```
5. **运行程序**：按 F5

### 方法2：使用命令行

```bash
cd WPF_Classifier_Demo
dotnet build -c Debug
copy ..\CPP\build\bin\Release\*.dll bin\x64\Debug\net6.0-windows\
copy ..\CPP\build\bin\Release\*.onnx bin\x64\Debug\net6.0-windows\
copy ..\CPP\build\bin\Release\*.json bin\x64\Debug\net6.0-windows\
dotnet run
```

---

## 🎯 功能说明

### 1. 单样本预测
- 使用内置示例数据
- 显示预测类别、置信度和概率分布

### 2. 批量 CSV 预测
- 选择 CSV 文件
- 批量处理所有样本
- 显示每个样本的结果

---

## 📝 修改代码后重新编译

### 修改 C# 代码
直接在 Visual Studio 中按 F5，会自动重新编译。

### 修改 C++ DLL
```bash
# 1. 关闭 WPF 程序
# 2. 重新编译 DLL
cd ..\CPP
cmake --build build --config Release
# 3. 复制新 DLL
copy build\bin\Release\ClassifierDLL.dll ..\WPF_Classifier_Demo\bin\x64\Debug\net6.0-windows\
# 4. 重新运行 WPF 程序
```

**重要**：必须先关闭 WPF 程序，否则 DLL 被占用无法覆盖！

---

## ⚠️ 常见问题

### Q1: 提示"缺少模型文件"
**解决**：复制 DLL 和模型文件到程序目录

### Q2: 提示"初始化失败"
**解决**：确保使用最新的 DLL（已修复路径编码问题）

### Q3: 批量预测失败
**解决**：检查 CSV 格式（20个特征，逗号分隔，无表头）

### Q4: 复制 DLL 时提示"文件正在使用"
**解决**：先关闭 WPF 程序

---

## 📚 详细文档

查看主项目 README 获取完整文档：`../README.md`
