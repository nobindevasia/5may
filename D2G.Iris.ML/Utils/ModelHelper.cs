using System;
using System.Collections.Generic;
using System.IO;
using System.Threading.Tasks;
using Microsoft.ML;
using Microsoft.ML.Data;
using D2G.Iris.ML.Core.Enums;
using D2G.Iris.ML.Core.Interfaces;
using D2G.Iris.ML.Core.Models;

namespace D2G.Iris.ML.Utils
{
    public static class ModelHelper
    {
        public static async Task CreateModelInfo<TMetrics, TScore>(
            TMetrics metrics,
            IDataView dataView,
            string[] featureNames,
            ModelConfig config,
            ProcessedData processedData)
        {
            try
            {
                CreateModelInfoTxtFile(
                    null,
                    new GeneralInfo
                    {
                        AuthorName = config.Author,
                        Description = config.Description
                    },
                    new TrainingInfo
                    {

                        TrainerName = config.TrainingParameters.Algorithm,
                        NumberColumns = featureNames.Length + 1,
                        NumberRows = GetRowCount(dataView),
                        InputColumnNames = featureNames,
                        OutputColumnNames = new[] { config.TargetField }
                    },
                    ConvertToStandardizedMetrics(metrics, config.ModelType),
                    config,
                    processedData,
                    metrics); 

                await Task.CompletedTask;
            }
            catch (Exception ex)
            {
                Console.WriteLine($"Error saving model info: {ex.Message}");
            }
        }

        private static long GetRowCount(IDataView dataView)
        {
            return dataView.GetRowCount() ?? 0;
        }

        private static IStandardizedBaseMetrics ConvertToStandardizedMetrics<TMetrics>(TMetrics metrics, ModelType modelType)
        {
            switch (modelType)
            {
                case ModelType.BinaryClassification:
                    if (metrics is BinaryClassificationMetrics binaryMetrics)
                    {
                        return new StandardizedBinaryMetrics
                        {
                            Accuracy = binaryMetrics.Accuracy,
                            AreaUnderRocCurve = binaryMetrics.AreaUnderRocCurve,
                            PositivePrecision = binaryMetrics.PositivePrecision,
                            PositiveRecall = binaryMetrics.PositiveRecall,
                            F1Score = binaryMetrics.F1Score,
                            AreaUnderPrecisionRecallCurve = binaryMetrics.AreaUnderPrecisionRecallCurve
                        };
                    }
                    break;
                case ModelType.MultiClassClassification:
                    if (metrics is MulticlassClassificationMetrics multiclassMetrics)
                    {
                        return new StandardizedMulticlassMetrics
                        {
                            MicroAccuracy = multiclassMetrics.MicroAccuracy,
                            MacroAccuracy = multiclassMetrics.MacroAccuracy,
                            LogLoss = multiclassMetrics.LogLoss,
                            LogLossReduction = multiclassMetrics.LogLossReduction
                        };
                    }
                    break;
                case ModelType.Regression:
                    if (metrics is RegressionMetrics regressionMetrics)
                    {
                        return new StandardizedRegressionMetrics
                        {
                            RSquared = regressionMetrics.RSquared,
                            MeanAbsoluteError = regressionMetrics.MeanAbsoluteError,
                            MeanSquaredError = regressionMetrics.MeanSquaredError,
                            RootMeanSquaredError = regressionMetrics.RootMeanSquaredError
                        };
                    }
                    break;
            }

            return new DefaultStandardizedMetrics();
        }

        private static void CreateModelInfoTxtFile(
            string path,
            GeneralInfo generalInfo,
            TrainingInfo trainingInfo,
            IStandardizedBaseMetrics metricsInfo,
            ModelConfig config,
            ProcessedData processedData,
            object originalMetrics) 
        {
            if (path == null)
            {
                path = Directory.GetCurrentDirectory();
            }

            var filename = "ModelInfo.txt";
            using (StreamWriter writer = new StreamWriter(Path.Combine(path, filename)))
            {
                writer.WriteLine($"*********** MODEL INFORMATION ***********\n");
                writer.WriteLine($"Author name: {generalInfo.AuthorName}");
                writer.WriteLine($"Short description: {generalInfo.Description}\n");
                writer.WriteLine("-------------------------------------------------------");
                writer.WriteLine("Information about training-data:\n");
                writer.WriteLine($"Used Trainer/Algorithm: {trainingInfo.TrainerName}");
                writer.WriteLine("Dimensions:");
                writer.WriteLine($"Number of columns: {trainingInfo.NumberColumns}");
                writer.WriteLine($"Number of rows: {trainingInfo.NumberRows}\n");
                writer.WriteLine("Input column data information:");
                writer.WriteLine(string.Join("\n", trainingInfo.InputColumnNames));
                writer.WriteLine("\nOutput column data information:");
                writer.WriteLine(string.Join("\n", trainingInfo.OutputColumnNames));

                if (config != null && processedData != null)
                {
                    writer.WriteLine("\n-------------------------------------------------------");
                    writer.WriteLine("Data Balancing Information:");
                    writer.WriteLine($"Method: {config.DataBalancing.Method}");
                    writer.WriteLine($"Original Sample Count: {processedData.OriginalSampleCount}");
                    writer.WriteLine($"Balanced Sample Count: {processedData.BalancedSampleCount}");
                    if (config.DataBalancing.Method != DataBalanceMethod.None)
                    {
                        writer.WriteLine($"Undersampling Ratio: {config.DataBalancing.UndersamplingRatio:F2}");
                        writer.WriteLine($"Minority to Majority Ratio: {config.DataBalancing.MinorityToMajorityRatio:F2}");
                        writer.WriteLine($"K Neighbors: {config.DataBalancing.KNeighbors}");
                    }

                    writer.WriteLine("\n-------------------------------------------------------");
                    writer.WriteLine("Feature Selection Information:");
                    writer.WriteLine($"Method: {config.FeatureEngineering.Method}");

                    writer.WriteLine("\nSelected Features:");
                    foreach (var feature in processedData.FeatureNames)
                    {
                        writer.WriteLine($"  - {feature}");
                    }

                    switch (config.FeatureEngineering.Method)
                    {
                        case FeatureSelectionMethod.PCA:
                            writer.WriteLine($"Number of Components: {config.FeatureEngineering.NumberOfComponents}");
                            break;
                        case FeatureSelectionMethod.Correlation:
                            writer.WriteLine($"Multicollinearity Threshold: {config.FeatureEngineering.MulticollinearityThreshold:F2}");
                            writer.WriteLine($"Max Features: {config.FeatureEngineering.MaxFeatures}");
                            break;
                    }

                    if (!string.IsNullOrEmpty(processedData.FeatureSelectionReport))
                    {
                        writer.WriteLine("\nFeature Selection Report:");
                        writer.WriteLine(processedData.FeatureSelectionReport);
                    }
                }

                writer.WriteLine("\n-------------------------------------------------------");
                writer.WriteLine(metricsInfo.CreateStandardizedMetricsMsg());

               
                if (originalMetrics is BinaryClassificationMetrics binaryMetrics)
                {
                    writer.WriteLine("\n-------------------------------------------------------");
                    writer.WriteLine("Confusion Matrix:");
                    writer.WriteLine(binaryMetrics.ConfusionMatrix.GetFormattedConfusionTable());
                }
            
                else if (originalMetrics is MulticlassClassificationMetrics multiClassMetrics)
                {
                    writer.WriteLine("\n-------------------------------------------------------");
                    writer.WriteLine("Confusion Matrix:");
                    writer.WriteLine(multiClassMetrics.ConfusionMatrix.GetFormattedConfusionTable());
                }
            }

            Console.WriteLine($"Model info saved to {Path.Combine(path, filename)}");
        }
    }
}