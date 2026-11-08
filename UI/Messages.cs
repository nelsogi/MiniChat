using Spectre.Console;

namespace MiniChat.UI;

/// <summary>
/// Les messages de Nel Code, chacun dans sa couleur du Theme. Le texte est affiché tel quel :
/// les "[" et "]" d'un chemin ou d'une réponse du modèle ne sont jamais lus comme du markup Spectre.
/// Passe par AnsiConsole, donc s'affiche proprement au-dessus d'une barre de progression en cours.
/// </summary>
public static class Messages
{
    public static void Info(string text) => Line(Theme.Text, text);

    public static void Dim(string text) => Line(Theme.Dim, text);

    public static void Success(string text) => Line(Theme.Success, $"✓ {text}");

    public static void Warning(string text) => Line(Theme.Warning, $"! {text}");

    public static void Error(string text) => Line(Theme.Error, $"✗ {text}");

    /// <summary>Début d'une étape longue (une phase d'entraînement...) : une ligne avec un titre.</summary>
    public static void Title(string text)
    {
        AnsiConsole.WriteLine();
        AnsiConsole.Write(new Rule($"[{Theme.Accent}]{Markup.Escape(text)}[/]").LeftJustified().RuleStyle(Theme.Style(Theme.Border)));
    }

    /// <summary>Un bloc de texte encadré (le rapport des questions de test...).</summary>
    public static void Block(string header, string text)
    {
        AnsiConsole.Write(new Panel(new Text(text))
            .Header($"[{Theme.Accent}] {Markup.Escape(header)} [/]")
            .BorderStyle(Theme.Style(Theme.Border))
            .Expand());
    }

    /// <summary>Un spinner pendant un travail sans étapes à compter (lecture de texte.txt, tokenizer...).</summary>
    public static T Spin<T>(string status, Func<T> work) =>
        AnsiConsole.Status()
            .Spinner(Spinner.Known.Dots)
            .SpinnerStyle(Theme.Style(Theme.Accent))
            .Start(Markup.Escape(status), _ => work());

    /// <summary>"mm:ss", ou "h:mm:ss" à partir d'une heure (un entraînement long peut durer des heures).</summary>
    public static string Duration(TimeSpan? time) =>
        time is not { } t ? "--:--"
        : t.TotalHours >= 1 ? $"{(int)t.TotalHours}:{t:mm\\:ss}"
        : t.ToString(@"mm\:ss");

    private static void Line(string color, string text) =>
        AnsiConsole.MarkupLine($"[{color}]{Markup.Escape(text)}[/]");
}
