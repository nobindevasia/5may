using System;
using Microsoft.ML;
using D2G.Iris.ML.Core.Enums;
using D2G.Iris.ML.Core.Interfaces;

namespace D2G.Iris.ML.Training
{
    public class ModelTrainerFactory
    {
        private readonly MLContext _mlContext;
        private readonly TrainerFactory _trainerFactory;

        public ModelTrainerFactory(MLContext mlContext)
        {
            _mlContext = mlContext;
            _trainerFactory = new TrainerFactory(mlContext);
        }

        public IModelTrainer CreateTrainer(ModelType modelType)
        {
            return modelType switch
            {
                ModelType.BinaryClassification => new BinaryClassificationTrainer(_mlContext, _trainerFactory),
                ModelType.MultiClassClassification => new MultiClassClassificationTrainer(_mlContext, _trainerFactory),
                ModelType.Regression => new RegressionTrainer(_mlContext, _trainerFactory),
                _ => throw new ArgumentException($"Unsupported model type: {modelType}")
            };
        }
    }
}