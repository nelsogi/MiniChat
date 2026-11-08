using System.Text;
using MiniChat.Commands;
using Spectre.Console;

namespace MiniChat.UI;

/// <summary>
/// La zone de saisie, lue touche par touche :
///   ────────────────────────────────────────────
///   > /t_
///   ────────────────────────────────────────────
///   › /temp    Température : plus bas = ...       <- commandes qui commencent par ce qui est tapé
///     /test    Pose les questions de ...
///     ↑↓ choisir · Tab compléter · Entrée lancer · Échap fermer
///
/// Touches : ←→ Début Fin Retour arrière Suppr pour éditer ; Entrée pour envoyer.
/// Liste ouverte : ↑↓ choisissent, Tab complète, Entrée lance la commande choisie, Échap ferme la liste.
/// Liste fermée : ↑↓ parcourent l'historique, Échap vide la ligne, Ctrl+D sur une ligne vide quitte.
/// Une ligne trop longue défile à l'horizontale. Une fois envoyée, la zone est effacée :
/// le message est réaffiché dans la conversation (ChatView).
/// </summary>
public sealed class InputBox
{
    private const int MaxSuggestions = 6;

    private readonly IReadOnlyList<ICommand> _commands;
    private readonly List<string> _history = new();

    // La ligne en cours : texte, curseur, et premier caractère affiché (défilement horizontal).
    private readonly StringBuilder _text = new();
    private int _cursor;
    private int _scroll;

    // Historique : _historyIndex == _history.Count = la ligne en cours d'écriture, gardée dans _draft.
    private int _historyIndex;
    private string _draft = "";

    // Liste des commandes : celle choisie, et fermée par Échap (rouverte dès que le texte change).
    private int _selected;
    private bool _dismissed;

    // Lignes de la console occupées par la zone : _reserved lignes à partir de _top.
    private int _top;
    private int _reserved;

    public InputBox(IEnumerable<ICommand> commands) => _commands = commands.ToList();

    /// <summary>La ligne tapée, ou null pour quitter (Ctrl+D, ou fin de l'entrée redirigée).</summary>
    public string? Read()
    {
        // Entrée ou sortie redirigée (script, fichier) : pas de touches ni de curseur, une simple ligne.
        if (Console.IsInputRedirected || Console.IsOutputRedirected)
            return ReadPlain();

        string? line;
        try
        {
            line = ReadKeys();
        }
        finally
        {
            AnsiConsole.Cursor.Show();
        }

        if (!string.IsNullOrWhiteSpace(line) && (_history.Count == 0 || _history[^1] != line))
            _history.Add(line);
        return line;
    }

    private string? ReadKeys()
    {
        _text.Clear();
        (_cursor, _scroll, _selected, _dismissed) = (0, 0, 0, false);
        (_historyIndex, _draft) = (_history.Count, "");
        (_top, _reserved) = (Console.CursorTop, 1);
        Render();

        while (true)
        {
            var key = Console.ReadKey(intercept: true);
            var matches = Matches();

            switch (key.Key)
            {
                case ConsoleKey.Enter:
                    if (matches.Count > 0)
                        SetText("/" + matches[_selected].Name);
                    string line = _text.ToString();
                    Erase();
                    return line;

                case ConsoleKey.Tab:
                    if (matches.Count > 0)
                        SetText("/" + matches[_selected].Name + " ");
                    break;

                case ConsoleKey.UpArrow when matches.Count > 0:
                    _selected = (_selected - 1 + matches.Count) % matches.Count;
                    break;
                case ConsoleKey.DownArrow when matches.Count > 0:
                    _selected = (_selected + 1) % matches.Count;
                    break;
                case ConsoleKey.UpArrow:
                    MoveInHistory(-1);
                    break;
                case ConsoleKey.DownArrow:
                    MoveInHistory(+1);
                    break;

                case ConsoleKey.Escape:
                    if (matches.Count > 0)
                        _dismissed = true;
                    else
                        Edit(() => _text.Clear());
                    break;

                case ConsoleKey.Backspace when _cursor > 0:
                    Edit(() => _text.Remove(--_cursor, 1));
                    break;
                case ConsoleKey.Delete when _cursor < _text.Length:
                    Edit(() => _text.Remove(_cursor, 1));
                    break;
                case ConsoleKey.LeftArrow:
                    _cursor = Math.Max(0, _cursor - 1);
                    break;
                case ConsoleKey.RightArrow:
                    _cursor = Math.Min(_text.Length, _cursor + 1);
                    break;
                case ConsoleKey.Home:
                    _cursor = 0;
                    break;
                case ConsoleKey.End:
                    _cursor = _text.Length;
                    break;

                default:
                    if (key.KeyChar == '\u0004' && _text.Length == 0)   // Ctrl+D
                    {
                        Erase();
                        return null;
                    }
                    // AltGr donne aussi des caractères (@, #, {...) : tout ce qui n'est pas un caractère de contrôle s'écrit.
                    if (!char.IsControl(key.KeyChar))
                        Edit(() => _text.Insert(_cursor++, key.KeyChar));
                    break;
            }

            // Un collage arrive comme une rafale de touches : on ne redessine qu'une fois la rafale lue.
            if (!Console.KeyAvailable)
                Render();
        }
    }

    /// <summary>
    /// Les commandes à proposer : le texte commence par "/" et on tape encore le nom (pas d'espace).
    /// Vide si la liste est fermée.
    /// </summary>
    private List<ICommand> Matches()
    {
        string text = _text.ToString();
        if (_dismissed || !text.StartsWith('/') || text.Contains(' '))
            return new List<ICommand>();

        var matches = _commands.Where(c => c.Name.StartsWith(text[1..], StringComparison.OrdinalIgnoreCase)).ToList();
        _selected = Math.Clamp(_selected, 0, Math.Max(0, matches.Count - 1));
        return matches;
    }

    /// <summary>Une modification tapée : la liste se rouvre, sur la commande au nom exact s'il y en a une.</summary>
    private void Edit(Action change)
    {
        change();
        _dismissed = false;
        string name = _text.ToString().TrimStart('/');
        var matches = Matches();
        _selected = Math.Max(0, matches.FindIndex(c => c.Name.Equals(name, StringComparison.OrdinalIgnoreCase)));
    }

    private void SetText(string text)
    {
        _text.Clear().Append(text);
        _cursor = text.Length;
    }

    /// <summary>
    /// ↑ (delta -1) ou ↓ (+1) dans l'historique. La ligne en cours d'écriture est gardée et revient en bas.
    /// La liste reste fermée sur une ligne rappelée : ↑↓ continuent dans l'historique, même sur "/train".
    /// </summary>
    private void MoveInHistory(int delta)
    {
        int index = Math.Clamp(_historyIndex + delta, 0, _history.Count);
        if (index == _historyIndex)
            return;

        if (_historyIndex == _history.Count)
            _draft = _text.ToString();
        _historyIndex = index;
        SetText(index == _history.Count ? _draft : _history[index]);
        _dismissed = true;
    }

    // ---------------------------------------------------------------------------

    /// <summary>Redessine toute la zone, puis remet le curseur dans la ligne de saisie.</summary>
    private void Render()
    {
        int width = Math.Max(20, Console.BufferWidth);
        int available = width - 1 - (Theme.Prompt.Length + 1);   // largeur pour le texte, après "> "

        // Le curseur reste visible : le texte défile pour qu'il soit toujours dans la fenêtre.
        if (_cursor < _scroll) _scroll = _cursor;
        if (_cursor > _scroll + available) _scroll = _cursor - available;

        var matches = Matches();
        var lines = new List<string> { RuleLine(width), InputLine(available), RuleLine(width) };
        if (matches.Count > 0)
            lines.AddRange(SuggestionLines(matches, width));
        else
            lines.Add($"  [{Theme.Command}]/[/] [{Theme.Dim}]pour les commandes · ↑↓ historique ·[/] " +
                      $"[{Theme.Command}]/quit[/] [{Theme.Dim}]pour quitter[/]");

        AnsiConsole.Cursor.Hide();
        Reserve(lines.Count);
        string blank = new(' ', width - 1);
        for (int i = 0; i < _reserved; i++)
        {
            Console.SetCursorPosition(0, _top + i);
            Console.Write(blank);
            Console.SetCursorPosition(0, _top + i);
            if (i < lines.Count)
                AnsiConsole.Markup(lines[i]);
        }
        Console.SetCursorPosition(Theme.Prompt.Length + 1 + _cursor - _scroll, _top + 1);
        AnsiConsole.Cursor.Show();
    }

    /// <summary>
    /// Agrandit la zone jusqu'à lines lignes (jamais plus petite : les lignes en trop sont juste vidées).
    /// En bas de la console, les retours à la ligne la font défiler : la zone remonte d'autant.
    /// </summary>
    private void Reserve(int lines)
    {
        if (lines <= _reserved)
            return;

        Console.SetCursorPosition(0, _top + _reserved - 1);
        for (int i = _reserved; i < lines; i++)
            Console.WriteLine();
        _top = Console.CursorTop - (lines - 1);
        _reserved = lines;
    }

    /// <summary>Vide la zone et met le curseur à son début : la conversation continue à cet endroit.</summary>
    private void Erase()
    {
        string blank = new(' ', Math.Max(20, Console.BufferWidth) - 1);
        for (int i = 0; i < _reserved; i++)
        {
            Console.SetCursorPosition(0, _top + i);
            Console.Write(blank);
        }
        Console.SetCursorPosition(0, _top);
    }

    private static string RuleLine(int width) => $"[{Theme.Border}]{new string('─', width - 1)}[/]";

    private string InputLine(int available)
    {
        string prompt = $"[{Theme.Accent}]{Theme.Prompt}[/] ";
        if (_text.Length == 0)
            return prompt + $"[{Theme.Dim}]{Markup.Escape(Cut("Écris un message, ou / pour les commandes", available))}[/]";

        string visible = _text.ToString(_scroll, Math.Min(available, _text.Length - _scroll));
        return prompt + Markup.Escape(visible);
    }

    /// <summary>Une ligne par commande (au plus MaxSuggestions, la fenêtre suit la sélection), puis l'aide des touches.</summary>
    private IEnumerable<string> SuggestionLines(List<ICommand> matches, int width)
    {
        int nameWidth = matches.Max(c => c.Name.Length) + 1;
        int first = Math.Clamp(_selected - MaxSuggestions + 1, 0, Math.Max(0, matches.Count - MaxSuggestions));

        for (int i = first; i < Math.Min(matches.Count, first + MaxSuggestions); i++)
        {
            var command = matches[i];
            bool selected = i == _selected;
            string name = ("/" + command.Name).PadRight(nameWidth);
            string description = Cut(command.Description, width - 1 - (4 + nameWidth + 2));
            yield return selected
                ? $"[{Theme.Accent}]›[/] [bold {Theme.Accent}]{Markup.Escape(name)}[/]  {Markup.Escape(description)}"
                : $"  [{Theme.Command}]{Markup.Escape(name)}[/]  [{Theme.Dim}]{Markup.Escape(description)}[/]";
        }

        string more = matches.Count > MaxSuggestions ? $"{matches.Count} commandes · " : "";
        yield return $"  [{Theme.Dim}]{Markup.Escape(Cut($"{more}↑↓ choisir · Tab compléter · Entrée lancer · Échap fermer", width - 3))}[/]";
    }

    private static string Cut(string text, int length) =>
        length <= 1 ? "" : text.Length <= length ? text : text[..(length - 1)] + "…";

    private static string? ReadPlain()
    {
        // Entrée redirigée : rien n'est tapé à l'écran, ChatView réaffiche la ligne lue avec son ">".
        if (!Console.IsInputRedirected)
            AnsiConsole.Markup($"[{Theme.Accent}]{Theme.Prompt}[/] ");
        return Console.ReadLine();
    }
}
