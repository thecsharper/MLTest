using System;
using System.IO;
using Microsoft.ML;
using Microsoft.ML.Trainers;

namespace MLTest
{
    public class Program
    {
        // Example of ML logistic regression
        // https://visualstudiomagazine.com/articles/2019/10/17/logistic-regression-mlnet.aspx?oly_enc_id=7076I8140445I2L

        public static void Main()
        {
            Console.WriteLine("Begin ML.NET predict gender demo");
            var mlc = new MLContext(seed: 1);

            // 1. load data and create data pipeline
            Console.WriteLine("Loading normalized data into memory ");

            var trainDataPath = "employees_norm_train.tsv";

            Console.WriteLine(trainDataPath);

            var trainData = mlc.Data.LoadFromTextFile<ModelInput>(trainDataPath, '\t', hasHeader: true);

            var jobCategory = mlc.Transforms.Categorical.OneHotEncoding(new[] { new InputOutputColumnPair("job", "job") });
            var satisfactionCategory = mlc.Transforms.Categorical.OneHotEncoding(new[]{ new InputOutputColumnPair("satisfac", "satisfac") });
            
            var featureSet = mlc.Transforms.Concatenate("Features", new[] { "age", "job", "income", "satisfac" });
            var dataPipe = jobCategory.Append(satisfactionCategory).Append(featureSet);

            // 2. train model
            Console.WriteLine("Creating a logistic regression model");
            
            var options =
              new LbfgsLogisticRegressionBinaryTrainer.Options()
              {
                  LabelColumnName = "isMale",
                  FeatureColumnName = "Features",
                  MaximumNumberOfIterations = 100,
                  OptimizationTolerance = 1e-8f
              };

            var trainer = mlc.BinaryClassification.Trainers.LbfgsLogisticRegression(options);

            var trainPipe = dataPipe.Append(trainer);

            Console.WriteLine("Starting training");
            var model = trainPipe.Fit(trainData);

            Console.WriteLine("Training complete");

            // 3. evaluate model
            var predictions = model.Transform(trainData);
            var metrics = mlc.BinaryClassification.EvaluateNonCalibrated(predictions, "isMale", "Score");
            
            Console.Write("Model accuracy on training data = ");
            Console.WriteLine(metrics.Accuracy.ToString("F4"));

            // 4. use model
            var input = new ModelInput();
            input.Age = 0.32f; input.Job = "mgmt"; input.Income = 0.4900f;
            input.Satisfac = "medium";

            var pe = mlc.Model.CreatePredictionEngine<ModelInput, ModelOutput>(model);
            var Y = pe.Predict(input);
            
            Console.Write("Set age = 32, job = mgmt, income = $49K,");
            Console.WriteLine("satisfac = medium");
            Console.Write("Predicted isMale : ");
            Console.WriteLine(Y.PredictedLabel);

            Console.WriteLine("End ML.NET demo ");
            Console.ReadLine();
        }
    }
}
