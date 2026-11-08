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
            LossChart.Show(PhaseTitle(phase), points);
    }

    private static string PhaseTitle(string phase) => phase switch
    {
        "pretrain" => "Phase 1 - pré-entraînement",
        "finetune" => "Phase 2 - fine-tuning",
        _ => phase,
    };
}
