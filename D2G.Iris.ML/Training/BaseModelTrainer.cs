﻿using System;
using System.Linq;
using System.Threading.Tasks;
using Microsoft.ML;
using Microsoft.ML.Data;
using D2G.Iris.ML.Core.Interfaces;
using D2G.Iris.ML.Core.Models;
using D2G.Iris.ML.Utils;
using static Microsoft.ML.DataOperationsCatalog;

namespace D2G.Iris.ML.Training
{
    public abstract class BaseModelTrainer : IModelTrainer
    {
        protected readonly MLContext _mlContext;
        protected readonly TrainerFactory _trainerFactory;

        protected BaseModelTrainer(MLContext mlContext, TrainerFactory trainerFactory)
        {
            _mlContext = mlContext;
            _trainerFactory = trainerFactory;
        }

        public abstract Task<ITransformer> TrainModel(
            MLContext mlContext,
            IDataView dataView,
            string[] featureNames,
            ModelConfig config,
            ProcessedData processedData);


        protected IEstimator<ITransformer> GetBasePipeline(MLContext mlContext)
        {
            return mlContext.Transforms.NormalizeMinMax("Features")
                           .AppendCacheCheckpoint(mlContext);
        }

        protected TrainTestData SplitTrainTestData(
            MLContext mlContext,
            IDataView dataView,
            double testFraction)
        {
            Console.WriteLine($"Splitting data into training and testing sets (Test Fraction: {testFraction:P0})");
            return mlContext.Data.TrainTestSplit(
                dataView,
                testFraction: testFraction,
                seed: 42);
        }

        protected async Task<ITransformer> TrainModelAsync(
            IEstimator<ITransformer> pipeline,
            IDataView trainData)
        {
            Console.WriteLine("Starting model training...");
            var start = DateTime.Now;
            var model = await Task.Run(() => pipeline.Fit(trainData));
            var trainingTime = DateTime.Now - start;
            Console.WriteLine($"Training completed in {trainingTime.TotalSeconds:N1} seconds");
            return model;
        }

        protected void SaveModel(
            MLContext mlContext,
            ITransformer model,
            IDataView dataView,
            string modelPrefix,
            string algorithmName)
        {
            var modelPath = $"{modelPrefix}_{algorithmName}_Model.zip";
            mlContext.Model.Save(model, dataView.Schema, modelPath);
            Console.WriteLine($"Model saved: {modelPath}");
        }


        protected async Task SaveModelInfo<TMetrics>(
            TMetrics metrics,
            IDataView dataView,
            string[] featureNames,
            ModelConfig config,
            ProcessedData processedData) where TMetrics : class
        {
            await ModelHelper.CreateModelInfo<TMetrics, float>(
                metrics,
                dataView,
                featureNames,
                config,
                processedData);
        }

        protected BinaryClassificationMetrics EvaluateBinaryClassification(
            MLContext mlContext,
            ITransformer model,
            IDataView testData,
            string algorithmName)
        {
            Console.WriteLine("Evaluating binary classification model...");
            var predictions = model.Transform(testData);
            var metrics = mlContext.BinaryClassification.Evaluate(
                data: predictions,
                labelColumnName: "Label",
                scoreColumnName: "Score",
                predictedLabelColumnName: "PredictedLabel");

            ConsoleHelper.PrintBinaryClassificationMetrics(algorithmName, metrics);
            Console.WriteLine($"Confusion Matrix:\n{metrics.ConfusionMatrix.GetFormattedConfusionTable()}");

            return metrics;
        }

        protected MulticlassClassificationMetrics EvaluateMultiClassClassification(
            MLContext mlContext,
            ITransformer model,
            IDataView testData,
            string algorithmName)
        {
            Console.WriteLine("Evaluating multiclass classification model...");
            var predictions = model.Transform(testData);
            var metrics = mlContext.MulticlassClassification.Evaluate(predictions);

            ConsoleHelper.PrintMultiClassClassificationMetrics(algorithmName, metrics);
            Console.WriteLine($"Confusion Matrix:\n{metrics.ConfusionMatrix.GetFormattedConfusionTable()}");

            return metrics;
        }

        protected RegressionMetrics EvaluateRegression(
            MLContext mlContext,
            ITransformer model,
            IDataView testData,
            string algorithmName)
        {
            Console.WriteLine("Evaluating regression model...");
            var predictions = model.Transform(testData);
            var metrics = mlContext.Regression.Evaluate(predictions);

            ConsoleHelper.PrintRegressionMetrics(algorithmName, metrics);

            return metrics;
        }
    }
}