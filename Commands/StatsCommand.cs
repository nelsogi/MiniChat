using MiniChat.App;
using MiniChat.Training;
using MiniChat.UI;

namespace MiniChat.Commands;

/// <summary>Les courbes de loss du dernier entraînement (results/&lt;date&gt;-loss.csv), une par phase.</summary>
public sealed class StatsCommand : ICommand
{
    public string Name => "stats";
    public string Description => "Courbes de loss (entraînement et vérification) du dernier entraînement";
    public string Usage => "/stats";

    public void Execute(CommandContext context, string[] args)
    {
        string path = context.Workbench.LatestLossCsv()
                      ?? throw new UserException("Pas encore de courbe : aucun fichier results/*-loss.csv. Lance /train.");

        var phases = LossCsv.Read(path);
        if (phases.Count == 0)
            throw new UserException($"Aucune mesure dans {Path.GetFileName(path)} (l'entraînement vient de commencer ?).");

        Messages.Dim($"Fichier : {path}");
        foreach (var (phase, points) in phases)
        {
            LossChart.Show(PhaseTitle(phase), points);
            ShowPerCharacter(points);
        }
    }

    // La loss de vérification par caractère : au meilleur pas (celui des poids gardés) et à la dernière mesure.
    // Elle ne dépend pas du tokenizer : deux entraînements aux vocabulaires différents se comparent avec elle.
    private static void ShowPerCharacter(IReadOnlyList<LossPoint> points)
    {
        var measured = points.Where(p => p.ValLossPerChar is not null).ToList();
        if (measured.Count == 0)
        {
            Messages.Dim("  Loss par caractère : pas mesurée (entraînement d'avant cette mesure).");
            return;
        }

        var best = measured.MinBy(p => p.ValLoss)!;
        var last = measured[^1];
        Messages.Dim($"  Loss de vérification par caractère : {best.ValLossPerChar:F3} au meilleur pas ({best.Step:N0}), " +
                     $"{last.ValLossPerChar:F3} à la dernière mesure ({last.Step:N0}).");
    }

    private static string PhaseTitle(string phase) => phase switch
    {
        "pretrain" => "Phase 1 - pré-entraînement",
        "finetune" => "Phase 2 - fine-tuning",
        _ => phase,
    };
}
