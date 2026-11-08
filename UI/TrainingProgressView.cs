using System.Globalization;
using MiniChat.Training;
using Spectre.Console;

namespace MiniChat.UI;

/// <summary>
/// La barre de progression d'une phase d'entraînement :
///   ⠹ ━━━━━━━━━━━━━━╺━━━━━━━━━━  42%  pas 3 800/9 000 · loss 2.851 / 2.903 · lr 8.2e-4 · reste 39:52
/// Chaque mesure des loss laisse aussi une ligne au-dessus de la barre : l'historique reste à l'écran.
/// </summary>
public sealed class TrainingProgressView : ITrainingProgress
{
    private readonly ProgressTask _task;
    private readonly int _firstStep;
    private readonly int _lastStep;
    private TrainingReport? _lastReport;

    private TrainingProgressView(ProgressTask task, int firstStep, int lastStep)
    {
        _task = task;
        _firstStep = firstStep;
        _lastStep = lastStep;
    }

    /// <summary>Lance work avec une barre qui va de firstStep à lastStep, et renvoie son résultat.</summary>
    public static T Run<T>(int firstStep, int lastStep, Func<ITrainingProgress, T> work) =>
        AnsiConsole.Progress()
            .AutoClear(false)
            .HideCompleted(false)
            .Columns(
                new SpinnerColumn(Spinner.Known.Dots) { Style = Theme.Style(Theme.Accent) },
                new ProgressBarColumn
                {
                    Width = 30,
                    CompletedStyle = Theme.Style(Theme.Accent),
                    FinishedStyle = Theme.Style(Theme.Success),
                    RemainingStyle = Theme.Style(Theme.Border),
                },
                new PercentageColumn { Style = Theme.Style(Theme.Text), CompletedStyle = Theme.Style(Theme.Success) },
                new TaskDescriptionColumn { Alignment = Justify.Left })
            .Start(context =>
            {
                var task = context.AddTask("démarrage...", maxValue: Math.Max(1, lastStep - firstStep));
                var view = new TrainingProgressView(task, firstStep, lastStep);
                T result = work(view);
                task.StopTask();
                return result;
            });

    public void Advance(int step, double learningRate, TimeSpan? remaining)
    {
        _task.Value = step - _firstStep;
        string losses = _lastReport is { } r
            ? $"loss [{Theme.TrainLoss}]{r.TrainLoss:F3}[/] / [{Theme.ValLoss}]{r.ValLoss:F3}[/]"
            : $"[{Theme.Dim}]loss en attente[/]";
        _task.Description = $"pas {step:N0}/{_lastStep:N0} [{Theme.Dim}]·[/] {losses} [{Theme.Dim}]·[/] " +
                            $"lr {Number(learningRate)} [{Theme.Dim}]·[/] reste {Messages.Duration(remaining)}";
    }

    public void Report(TrainingReport report)
    {
        _lastReport = report;
        // Le temps restant est sur la barre : la ligne d'historique garde le temps écoulé.
        string gpu = report.Gpu is var (used, _) ? $" │ GPU {used:N0} Mo" : "";
        string best = report.IsBest ? $"  [{Theme.Best}]★ meilleur[/]" : "";
        AnsiConsole.MarkupLine(
            $"[{Theme.Dim}]  pas {report.Step,6:N0}/{report.LastStep:N0} │[/] " +
            $"entraînement [{Theme.TrainLoss}]{report.TrainLoss:F3}[/] [{Theme.Dim}]│[/] " +
            $"vérification [{Theme.ValLoss}]{report.ValLoss:F3}[/] " +
            $"[{Theme.Dim}]│ lr {Number(report.LearningRate)}{gpu} │ {Messages.Duration(report.Elapsed)}[/]{best}");
        Advance(report.Step, report.LearningRate, report.Remaining);
    }

    public void Log(string message)
    {
        foreach (string line in message.Split('\n'))
            Messages.Dim($"  {line}");
    }

    // "3.0e-4" : toujours un point, la vitesse s'écrit comme dans --lr.
    private static string Number(double value) => value.ToString("0.0e0", CultureInfo.InvariantCulture);
}
