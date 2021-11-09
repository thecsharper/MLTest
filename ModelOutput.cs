using Microsoft.ML.Data;

namespace MLTest
{
    public class ModelOutput
    {
        [ColumnName("predictedLabel")]
        public bool PredictedLabel { get; set; }

        [ColumnName("score")]
        public float Score { get; set; }
    }
}
