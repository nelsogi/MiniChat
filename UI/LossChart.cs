using System.Text;
using MiniChat.Training;
using Spectre.Console;

namespace MiniChat.UI;

/// <summary>
/// Les deux courbes de loss d'une phase, dessinées en caractères :
///   3.40 ┤•
///        │ ••
///        │   •••••
///   2.70 ┤ ••    ••••••••★•••••
///        └──────────────────────
///         0          pas         9 000
/// Entraînement et vérification ont chacune leur couleur ; ★ = la meilleure loss de vérification (les poids gardés).
/// </summary>
public static class LossChart
{
    private const int Height = 14;
    private const int MaxWidth = 100;
    private const int LabelWidth = 6;   // "12.34 " : les valeurs de l'axe vertical

    private const char TrainMark = '·';
    private const char ValMark = '•';
    private const char BestMark = '★';

    public static void Show(string title, IReadOnlyList<LossPoint> points)
    {
        Messages.Title(title);
        if (points.Count == 0)
        {
            Messages.Dim("(aucune mesure)");
            return;
        }

        int width = Math.Clamp(AnsiConsole.Profile.Width - LabelWidth - 3, 10, MaxWidth);
        int firstStep = points[0].Step;
        int lastStep = points[^1].Step;
        var best = points.MinBy(p => p.ValLoss)!;

        // Au tout début la loss est très haute (modèle au hasard) : avec elle dans l'échelle, la suite serait
        // écrasée en bas. Le haut de l'échelle ignore donc le premier dixième des mesures ; ce qui dépasse
        // est dessiné sur la ligne du haut.
        var scaled = points.Count >= 10 ? points.Skip(points.Count / 10).ToList() : points;
        float top = scaled.Max(p => Math.Max(p.TrainLoss, p.ValLoss));
        float bottom = points.Min(p => Math.Min(p.TrainLoss, p.ValLoss));
        float peak = points.Max(p => Math.Max(p.TrainLoss, p.ValLoss));
        float margin = Math.Max((top - bottom) * 0.05f, 0.01f);
        (top, bottom) = (top + margin, Math.Max(0, bottom - margin));

        var cells = new char[Height, width];
        var colors = new string[Height, width];
        for (int row = 0; row < Height; row++)
            for (int column = 0; column < width; column++)
                cells[row, column] = ' ';

        // Une valeur par colonne, interpolée entre les mesures voisines : la courbe reste continue même
        // avec peu de mesures (fine-tuning : une tous les 25 pas). Vérification dessinée en dernier : devant.
        for (int column = 0; column < width; column++)
        {
            double step = width == 1 || lastStep == firstStep
                ? firstStep
                : firstStep + (lastStep - firstStep) * (double)column / (width - 1);
            Plot(cells, colors, column, Row(At(points, step, p => p.TrainLoss), top, bottom), TrainMark, Theme.TrainLoss);
            Plot(cells, colors, column, Row(At(points, step, p => p.ValLoss), top, bottom), ValMark, Theme.ValLoss);
        }
        Plot(cells, colors, Column(best.Step, firstStep, lastStep, width), Row(best.ValLoss, top, bottom), BestMark, Theme.Best);

        var chart = new StringBuilder();
        for (int row = 0; row < Height; row++)
        {
            // Une valeur sur l'axe toutes les 4 lignes, et sur la dernière.
            bool labeled = row % 4 == 0 || row == Height - 1;
            float value = top - (top - bottom) * row / (Height - 1);
            chart.Append($"[{Theme.Dim}]{(labeled ? value.ToString("0.00") : "").PadLeft(LabelWidth)} {(labeled ? '┤' : '│')}[/]");
            for (int column = 0; column < width; column++)
                chart.Append(colors[row, column] is { } color ? $"[{color}]{cells[row, column]}[/]" : " ");
            chart.Append('\n');
        }

        string first = firstStep.ToString("N0");
        string last = lastStep.ToString("N0");
        string middle = "pas";
        int gap = Math.Max(1, width - first.Length - last.Length - middle.Length);
        chart.Append($"[{Theme.Dim}]{new string(' ', LabelWidth)} └{new string('─', width)}[/]\n");
        chart.Append($"[{Theme.Dim}]{new string(' ', LabelWidth + 2)}{first}{new string(' ', gap / 2)}{middle}" +
                     $"{new string(' ', gap - gap / 2)}{last}[/]");
        AnsiConsole.MarkupLine(chart.ToString());

        var end = points[^1];
        AnsiConsole.MarkupLine(
            $"  [{Theme.TrainLoss}]{TrainMark} entraînement[/]  [{Theme.ValLoss}]{ValMark} vérification[/]  " +
            $"[{Theme.Best}]{BestMark} meilleure : {best.ValLoss:F3} au pas {best.Step:N0}[/]");
        Messages.Dim($"  {points.Count} mesures · dernière au pas {end.Step:N0} : entraînement {end.TrainLoss:F3}, " +
                     $"vérification {end.ValLoss:F3}" + (peak > top ? $" · début coupé (loss jusqu'à {peak:F2})" : ""));
    }

    /// <summary>La loss au pas step, en ligne droite entre les deux mesures qui l'entourent.</summary>
    private static float At(IReadOnlyList<LossPoint> points, double step, Func<LossPoint, float> loss)
    {
        int after = 0;
        while (after < points.Count - 1 && points[after].Step < step)
            after++;
        if (after == 0)
            return loss(points[0]);

        var (a, b) = (points[after - 1], points[after]);
        double t = (step - a.Step) / Math.Max(1, b.Step - a.Step);
        return (float)(loss(a) + (loss(b) - loss(a)) * t);
    }

    private static int Row(float value, float top, float bottom) =>
        Math.Clamp((int)Math.Round((top - value) / (top - bottom) * (Height - 1)), 0, Height - 1);

    private static int Column(int step, int firstStep, int lastStep, int width) =>
        lastStep == firstStep ? 0 : (int)Math.Round((double)(step - firstStep) / (lastStep - firstStep) * (width - 1));

    private static void Plot(char[,] cells, string[,] colors, int column, int row, char mark, string color)
    {
        cells[row, column] = mark;
        colors[row, column] = color;
    }
}
