using System;
using System.Linq;
using System.Threading.Tasks;
using System.Collections.Generic;
using Microsoft.ML;
using Microsoft.ML.Data;
using D2G.Iris.ML.Core.Interfaces;
using D2G.Iris.ML.Core.Models;
using D2G.Iris.ML.Utils;

namespace D2G.Iris.ML.Training
{
    public class BinaryClassificationTrainer : BaseModelTrainer
    {
        public BinaryClassificationTrainer(MLContext mlContext, TrainerFactory trainerFactory)
            : base(mlContext, trainerFactory)
        {
        }

        public override async Task<ITransformer> TrainModel(
            MLContext mlContext,
            IDataView dataView,
            string[] featureNames,
            ModelConfig config,
            ProcessedData processedData)
        {
            Console.WriteLine($"\nStarting binary classification using {config.TrainingParameters.Algorithm}...");

            var labelPipeline = _mlContext.Transforms.CopyColumns(
                    outputColumnName: "RawLabel", inputColumnName: config.TargetField)
                .Append(_mlContext.Transforms.Conversion.ConvertType(
                    outputColumnName: "Label", inputColumnName: "RawLabel", outputKind: DataKind.Boolean));

            var labeledData = labelPipeline.Fit(dataView).Transform(dataView);

            IDataView fixedData = PrepareData(labeledData, featureNames);

            var split = SplitTrainTestData(
                _mlContext,
                fixedData,
                config.TrainingParameters.TestFraction);

            var trainer = _trainerFactory.GetTrainer(
                config.ModelType,
                config.TrainingParameters);

            var pipeline = GetBasePipeline(_mlContext)
                .Append(trainer)
                .Append(_mlContext.Transforms.CopyColumns("Probability", "Score"));

            var model = await TrainModelAsync(pipeline, split.TrainSet);

            var metrics = EvaluateBinaryClassification(
                _mlContext,
                model,
                split.TestSet,
                config.TrainingParameters.Algorithm);

            await SaveModelInfo(
                metrics,
                fixedData,
                featureNames,
                config,
                processedData);

            SaveModel(
                _mlContext,
                model,
                fixedData,
                "BinaryClassification",
                config.TrainingParameters.Algorithm);

            return model;
        }

        private IDataView PrepareData(IDataView labeledData, string[] featureNames)
        {
            if (labeledData.Schema.GetColumnOrNull("Features").HasValue)
            {
                var temp = labeledData.GetColumn<VBuffer<float>>("Features")
                    .Zip(labeledData.GetColumn<bool>("Label"), (feat, lbl) => new BinaryVector { Features = feat.GetValues().ToArray(), Label = lbl })
                    .ToList();

                var schemaDef = SchemaDefinition.Create(typeof(BinaryVector));
                schemaDef[nameof(BinaryVector.Features)].ColumnType =
                    new VectorDataViewType(NumberDataViewType.Single, featureNames.Length);

                return _mlContext.Data.LoadFromEnumerable(temp, schemaDef);
            }
            else
            {
                return _mlContext.Transforms.Concatenate("Features", featureNames)
                    .Fit(labeledData)
                    .Transform(labeledData);
            }
        }

        private class BinaryVector
        {
            [VectorType]
            public float[] Features { get; set; }
            public bool Label { get; set; }
        }
    }
}