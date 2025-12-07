using System;
using System.Windows;
using OpticalClassifier;

namespace WPFClassifierExample
{
    /// <summary>
    /// WPF 使用示例
    /// </summary>
    public partial class MainWindow : Window
    {
        private Classifier classifier;

        public MainWindow()
        {
            InitializeComponent();
            InitializeClassifier();
        }

        /// <summary>
        /// 初始化分类器
        /// </summary>
        private void InitializeClassifier()
        {
            try
            {
                classifier = new Classifier();

                // 初始化分类器（修改为你的实际路径）
                string modelPath = @"lightgbm_model.onnx";
                string scalerPath = @"scaler_params.json";
                string labelPath = @"label_mapping.json";

                bool success = classifier.Initialize(modelPath, scalerPath, labelPath);

                if (success)
                {
                    MessageBox.Show("分类器初始化成功！", "成功", MessageBoxButton.OK, MessageBoxImage.Information);
                }
            }
            catch (Exception ex)
            {
                MessageBox.Show($"初始化失败: {ex.Message}", "错误", MessageBoxButton.OK, MessageBoxImage.Error);
            }
        }

        /// <summary>
        /// 预测按钮点击事件
        /// </summary>
        private void PredictButton_Click(object sender, RoutedEventArgs e)
        {
            try
            {
                // 示例：从界面获取20个特征值
                float[] features = new float[20]
                {
                    581.722f, -0.411162f, -0.262966f, 0.1029f, 355.387f,
                    // ... 其他15个特征
                    // 这里需要从你的 UI 控件中获取实际数据
                    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
                };

                // 调用预测
                var result = classifier.Predict(features);

                // 显示结果
                string message = $"预测结果:\n\n" +
                                $"类别: {result.ClassName}\n" +
                                $"置信度: {result.Confidence:P2}\n\n" +
                                $"所有类别概率:\n";

                for (int i = 0; i < result.Probabilities.Length; i++)
                {
                    string className = classifier.GetClassName(i);
                    message += $"  {className}: {result.Probabilities[i]:P2}\n";
                }

                MessageBox.Show(message, "预测结果", MessageBoxButton.OK, MessageBoxImage.Information);
            }
            catch (Exception ex)
            {
                MessageBox.Show($"预测失败: {ex.Message}", "错误", MessageBoxButton.OK, MessageBoxImage.Error);
            }
        }

        /// <summary>
        /// 窗口关闭时释放资源
        /// </summary>
        protected override void OnClosed(EventArgs e)
        {
            classifier?.Dispose();
            base.OnClosed(e);
        }
    }

    /// <summary>
    /// 更完整的使用示例
    /// </summary>
    public class ClassifierExample
    {
        public static void Example1_BasicUsage()
        {
            // 创建分类器实例
            using (var classifier = new Classifier())
            {
                // 初始化
                classifier.Initialize(
                    @"lightgbm_model.onnx",
                    @"scaler_params.json",
                    @"label_mapping.json"
                );

                // 准备特征数据
                float[] features = new float[20] { /* 你的20个特征值 */ };

                // 预测
                var result = classifier.Predict(features);

                // 使用结果
                Console.WriteLine($"预测类别: {result.ClassName}");
                Console.WriteLine($"置信度: {result.Confidence:P2}");
            }
        }

        public static void Example2_BatchPrediction()
        {
            using (var classifier = new Classifier())
            {
                classifier.Initialize(
                    @"lightgbm_model.onnx",
                    @"scaler_params.json",
                    @"label_mapping.json"
                );

                // 批量预测多个样本
                float[][] samples = new float[][]
                {
                    new float[20] { /* 样本1 */ },
                    new float[20] { /* 样本2 */ },
                    new float[20] { /* 样本3 */ }
                };

                foreach (var sample in samples)
                {
                    var result = classifier.Predict(sample);
                    Console.WriteLine($"预测: {result.ClassName} ({result.Confidence:P2})");
                }
            }
        }

        public static void Example3_ErrorHandling()
        {
            try
            {
                using (var classifier = new Classifier())
                {
                    // 初始化可能失败
                    classifier.Initialize(
                        @"lightgbm_model.onnx",
                        @"scaler_params.json",
                        @"label_mapping.json"
                    );

                    float[] features = new float[20];

                    // 预测可能失败
                    var result = classifier.Predict(features);

                    Console.WriteLine($"成功: {result.ClassName}");
                }
            }
            catch (ArgumentException ex)
            {
                Console.WriteLine($"参数错误: {ex.Message}");
            }
            catch (InvalidOperationException ex)
            {
                Console.WriteLine($"操作错误: {ex.Message}");
            }
            catch (Exception ex)
            {
                Console.WriteLine($"未知错误: {ex.Message}");
            }
        }
    }
}
