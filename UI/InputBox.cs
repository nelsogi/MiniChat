using Spectre.Console;

namespace MiniChat.UI;

/// <summary>
/// La zone de saisie, entre deux lignes, avec une astuce dessous :
///   ────────────────────────────────
///   > ton message_
///   ────────────────────────────────
///     /help pour les commandes · /quit pour quitter
/// Une fois validée, la zone est effacée : le message est réaffiché dans la conversation (ChatView).
/// La saisie reste celle de la console (flèches, historique avec ↑) : Console.ReadLine.
/// </summary>
public static class InputBox
{
    private const int BoxLines = 4;   // ligne du haut, saisie, ligne du bas, astuce

    /// <summary>La ligne tapée, ou null si l'entrée est fermée (Ctrl+Z, fichier redirigé fini).</summary>
    public static string? Read()
    {
        // Entrée ou sortie redirigée (script, fichier) : pas de curseur à déplacer, une simple invite.
        if (Console.IsInputRedirected || Console.IsOutputRedirected)
            return ReadPlain();

        try
        {
            return ReadInBox();
        }
        catch (IOException)
        {
            return ReadPlain();   // console qui ne permet pas de déplacer le curseur
        }
    }

    private static string? ReadInBox()
    {
        var border = Theme.Style(Theme.Border);
        AnsiConsole.Write(new Rule().RuleStyle(border));
        AnsiConsole.WriteLine();
        AnsiConsole.Write(new Rule().RuleStyle(border));
        AnsiConsole.MarkupLine($"  [{Theme.Command}]/help[/] [{Theme.Dim}]pour les commandes ·[/] " +
                               $"[{Theme.Command}]/quit[/] [{Theme.Dim}]pour quitter[/]");

        // Lu après l'écriture : si la console a défilé pour faire de la place, la position en tient compte.
        int top = Console.CursorTop - BoxLines;
        Console.SetCursorPosition(0, top + 1);
        AnsiConsole.Markup($"[{Theme.Accent}]{Theme.Prompt}[/] ");
        string? input = Console.ReadLine();

        // Un long message passe sur plusieurs lignes ; en bas de l'écran, la console a pu défiler encore.
        int width = Math.Max(1, Console.BufferWidth);
        int inputLines = (Theme.Prompt.Length + 1 + (input?.Length ?? 0)) / width + 1;
        top = Math.Max(0, Math.Min(top, Console.CursorTop - inputLines - 1));
        int bottom = Math.Min(Math.Max(Console.CursorTop, top + BoxLines), Console.BufferHeight);

        string blank = new(' ', width - 1);
        for (int line = top; line < bottom; line++)
        {
            Console.SetCursorPosition(0, line);
            Console.Write(blank);
        }
        Console.SetCursorPosition(0, top);
        return input;
    }

    private static string? ReadPlain()
    {
        // Entrée redirigée : rien n'est tapé à l'écran, ChatView réaffiche la ligne lue avec son ">".
        if (!Console.IsInputRedirected)
            AnsiConsole.Markup($"[{Theme.Accent}]{Theme.Prompt}[/] ");
        return Console.ReadLine();
    }
}
