using System;
using System.Linq;
using System.Threading.Tasks;
using Microsoft.ML;
using Microsoft.ML.Data;
using D2G.Iris.ML.Core.Models;
using D2G.Iris.ML.Core.Interfaces;
using D2G.Iris.ML.Utils;

namespace D2G.Iris.ML.Training
{
    public class MultiClassClassificationTrainer : BaseModelTrainer
    {
        public MultiClassClassificationTrainer(MLContext mlContext, TrainerFactory trainerFactory)
            : base(mlContext, trainerFactory)
        {
        }

        private class ModelInput
        {
            [VectorType]
            public float[] Features { get; set; }
            public long Label { get; set; }
        }

        public override async Task<ITransformer> TrainModel(
            MLContext mlContext,
            IDataView dataView,
            string[] featureNames,
            ModelConfig config,
            ProcessedData processedData)
        {
            Console.WriteLine($"\nStarting multiclass classification model training using {config.TrainingParameters.Algorithm}...");

            try
            {
                IDataView fixedData = PrepareData(dataView, featureNames);

                var splitData = SplitTrainTestData(
                    _mlContext,
                    fixedData,
                    config.TrainingParameters.TestFraction);

                IEstimator<ITransformer> pipeline = _mlContext.Transforms
                    .NormalizeMinMax("Features")
                    .Append(_mlContext.Transforms.Conversion
                        .MapValueToKey(outputColumnName: "Label", inputColumnName: "Label"))
                    .AppendCacheCheckpoint(_mlContext);

                var trainer = _trainerFactory.GetTrainer(
                    config.ModelType,
                    config.TrainingParameters);

                pipeline = pipeline
                    .Append(trainer)
                    .Append(_mlContext.Transforms.Conversion
                        .MapKeyToValue("PredictedLabel", "PredictedLabel"));

                var model = await TrainModelAsync(pipeline, splitData.TrainSet);

                var metrics = EvaluateMultiClassClassification(
                    _mlContext,
                    model,
                    splitData.TestSet,
                    config.TrainingParameters.Algorithm);


                await SaveModelInfo(
                    metrics,
                    dataView,
                    featureNames,
                    config,
                    processedData);


                SaveModel(
                    _mlContext,
                    model,
                    fixedData,
                    "MultiClassClassification",
                    config.TrainingParameters.Algorithm);

                return model;
            }
            catch (Exception ex)
            {
                Console.WriteLine($"\nError during model training: {ex.Message}");
                Console.WriteLine($"Stack trace: {ex.StackTrace}");
                if (ex.InnerException != null)
                    Console.WriteLine($"Inner Exception: {ex.InnerException.Message}");
                throw;
            }
        }

        private IDataView PrepareData(IDataView dataView, string[] featureNames)
        {
            var data = _mlContext.Data
                .CreateEnumerable<ModelInput>(dataView, reuseRowObject: false)
                .Select(row => new ModelInput
                {
                    Features = row.Features,
                    Label = row.Label
                })
                .ToList();

            var schema = SchemaDefinition.Create(typeof(ModelInput));
            schema["Features"].ColumnType =
                new VectorDataViewType(NumberDataViewType.Single, featureNames.Length);

            return _mlContext.Data.LoadFromEnumerable(data, schema);
        }
    }
}