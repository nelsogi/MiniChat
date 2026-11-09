using System.Globalization;

namespace MiniChat.Training;

/// <summary>Une ligne du CSV des loss : une mesure pendant l'entraînement (voir Trainer.WriteCsvLine).</summary>
public sealed record LossPoint(string Phase, int Step, float TrainLoss, float ValLoss, double LearningRate);

/// <summary>
/// Relit le CSV des loss écrit par le Trainer (results/&lt;date&gt;-loss.csv) :
///   phase,etape,loss_entrainement,loss_verification,learning_rate,secondes,tokens_par_seconde,memoire_gpu_mo
/// (les CSV d'avant n'ont pas les deux dernières colonnes ; seules les cinq premières sont relues ici).
/// Les deux phases sont dans le même fichier, et leurs pas repartent chacun de 0.
/// </summary>
public static class LossCsv
{
    /// <summary>
    /// Les mesures de chaque phase, dans l'ordre du fichier, triées par pas.
    /// Après une reprise (--resume), les pas refaits depuis le checkpoint sont mesurés deux fois :
    /// la dernière mesure d'un pas est celle qui compte.
    /// </summary>
    public static IReadOnlyList<(string Phase, IReadOnlyList<LossPoint> Points)> Read(string path)
    {
        var phases = new List<(string Phase, Dictionary<int, LossPoint> Points)>();

        foreach (string line in File.ReadLines(path).Skip(1))
        {
            if (Parse(line) is not { } point)
                continue;

            int index = phases.FindIndex(p => p.Phase == point.Phase);
            if (index < 0)
            {
                phases.Add((point.Phase, new Dictionary<int, LossPoint>()));
                index = phases.Count - 1;
            }
            phases[index].Points[point.Step] = point;
        }

        return phases
            .Select(p => (p.Phase, (IReadOnlyList<LossPoint>)p.Points.Values.OrderBy(v => v.Step).ToList()))
            .ToList();
    }

    // Nombres avec un point (format international), comme à l'écriture. Ligne abîmée (coupure pendant l'écriture) : ignorée.
    private static LossPoint? Parse(string line)
    {
        string[] fields = line.Split(',');
        if (fields.Length < 5)
            return null;

        var culture = CultureInfo.InvariantCulture;
        return int.TryParse(fields[1], NumberStyles.Integer, culture, out int step)
               && float.TryParse(fields[2], NumberStyles.Float, culture, out float train)
               && float.TryParse(fields[3], NumberStyles.Float, culture, out float val)
               && double.TryParse(fields[4], NumberStyles.Float, culture, out double lr)
            ? new LossPoint(fields[0], step, train, val, lr)
            : null;
    }
}
