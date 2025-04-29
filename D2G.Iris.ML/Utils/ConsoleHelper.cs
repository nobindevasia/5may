using System;
using Microsoft.ML.Data;

namespace D2G.Iris.ML.Utils
{
    public static class ConsoleHelper
    {
        public static void PrintBinaryClassificationMetrics(string name, BinaryClassificationMetrics metrics)
        {
            Console.WriteLine($"************************************************************");
            Console.WriteLine($"*    Model quality metrics evaluation for {name}");
            Console.WriteLine($"*-----------------------------------------------------------");
            Console.WriteLine($"*    Accuracy:          {metrics.Accuracy:P2}");
            Console.WriteLine($"*    AUC:               {metrics.AreaUnderRocCurve:P2}");
            Console.WriteLine($"*    Precision:         {metrics.PositivePrecision:P2}");
            Console.WriteLine($"*    Recall:            {metrics.PositiveRecall:P2}");
            Console.WriteLine($"*    F1 Score:          {metrics.F1Score:P2}");
            Console.WriteLine($"*    Negative Precision:{metrics.NegativePrecision:P2}");
            Console.WriteLine($"*    Negative Recall:   {metrics.NegativeRecall:P2}");
            Console.WriteLine($"*    AreaUnderPrecisionRecallCurve: {metrics.AreaUnderPrecisionRecallCurve:P2}");
            Console.WriteLine($"************************************************************");
        }

        public static void PrintMultiClassClassificationMetrics(string name, MulticlassClassificationMetrics metrics)
        {
            Console.WriteLine($"************************************************************");
            Console.WriteLine($"*    Model quality metrics evaluation for {name}");
            Console.WriteLine($"*-----------------------------------------------------------");
            Console.WriteLine($"*    Macro Accuracy:    {metrics.MacroAccuracy:P2}");
            Console.WriteLine($"*    Micro Accuracy:    {metrics.MicroAccuracy:P2}");
            Console.WriteLine($"*    Log Loss:          {metrics.LogLoss:0.###}");
            Console.WriteLine($"*    Log Loss Reduction:{metrics.LogLossReduction:0.###}");
            Console.WriteLine($"*    Top K Accuracy:    {metrics.TopKAccuracy:0.###}");
            Console.WriteLine($"************************************************************");
        }

        public static void PrintRegressionMetrics(string name, RegressionMetrics metrics)
        {
            Console.WriteLine($"************************************************************");
            Console.WriteLine($"*    Model quality metrics evaluation for {name}");
            Console.WriteLine($"*-----------------------------------------------------------");
            Console.WriteLine($"*    R² Score:          {metrics.RSquared:0.###}");
            Console.WriteLine($"*    Mean Squared Error:{metrics.MeanSquaredError:0.###}");
            Console.WriteLine($"*    Mean Absolute Error:{metrics.MeanAbsoluteError:0.###}");
            Console.WriteLine($"*    Root Mean Squared Error: {metrics.RootMeanSquaredError:0.###}");
            Console.WriteLine($"************************************************************");
        }
    }
}