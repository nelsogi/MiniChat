using System.Collections.Concurrent;
using MiniChat.Chat;
using Spectre.Console;

namespace MiniChat.UI;

/// <summary>
/// La conversation à l'écran :
///   > Bonjour !                    <- le message de l'utilisateur
///   ● Bonjour, comment vas-tu ?    <- la réponse, écrite au fur et à mesure
/// Un spinner tourne tant que le modèle n'a pas écrit son premier caractère.
/// </summary>
public static class ChatView
{
    public static void ShowUser(string message)
    {
        AnsiConsole.MarkupLine($"[{Theme.Dim}]{Theme.Prompt}[/] [{Theme.User}]{Markup.Escape(message)}[/]");
    }

    /// <summary>Envoie message au modèle et affiche sa réponse caractère par caractère.</summary>
    public static void Reply(ChatSession chat, string message)
    {
        // La génération tourne à part et pousse ses caractères ici ; l'affichage les reprend dans l'ordre.
        // Comme ça le spinner (sur ce thread) peut tourner jusqu'au premier caractère, puis laisser la place.
        using var characters = new BlockingCollection<char>();
        var generation = Task.Run(() =>
        {
            try { chat.Send(message, characters.Add); }
            finally { characters.CompleteAdding(); }
        });

        char first = default;
        bool any = false;
        AnsiConsole.Status()
            .Spinner(Spinner.Known.Dots)
            .SpinnerStyle(Theme.Style(Theme.Accent))
            .Start($"[{Theme.Dim}]Réflexion...[/]", _ => any = characters.TryTake(out first, Timeout.Infinite));

        AnsiConsole.Markup($"[{Theme.Accent}]{Theme.ReplyBullet}[/] ");
        if (any)
        {
            Write(first);
            foreach (char c in characters.GetConsumingEnumerable())
                Write(c);
        }
        else
        {
            AnsiConsole.Markup($"[{Theme.Dim}](pas de réponse)[/]");
        }
        AnsiConsole.WriteLine();
        AnsiConsole.WriteLine();

        generation.GetAwaiter().GetResult();   // une erreur pendant la génération remonte ici
    }

    private static void Write(char c) => AnsiConsole.Write(new Text(c.ToString(), Theme.Style(Theme.Text)));
}
