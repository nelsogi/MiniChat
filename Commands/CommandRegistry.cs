using MiniChat.App;

namespace MiniChat.Commands;

/// <summary>
/// Toutes les commandes, trouvées dans le programme : chaque classe qui implémente ICommand
/// (avec un constructeur sans paramètre) est créée une fois, au démarrage.
/// </summary>
public sealed class CommandRegistry
{
    private readonly Dictionary<string, ICommand> _commands;

    public CommandRegistry()
    {
        _commands = typeof(ICommand).Assembly.GetTypes()
            .Where(type => type.IsClass && !type.IsAbstract && typeof(ICommand).IsAssignableFrom(type))
            .Select(type => (ICommand)Activator.CreateInstance(type)!)
            .ToDictionary(command => command.Name, StringComparer.OrdinalIgnoreCase);
    }

    /// <summary>Par ordre alphabétique (pour /help).</summary>
    public IEnumerable<ICommand> Commands => _commands.Values.OrderBy(command => command.Name, StringComparer.Ordinal);

    /// <summary>Exécute une ligne qui commence par "/", ex. "/temp 0.8".</summary>
    public void Execute(CommandContext context, string line)
    {
        string[] words = line.TrimStart('/').Split(' ', StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries);
        string name = words.Length > 0 ? words[0] : "";

        if (!_commands.TryGetValue(name, out var command))
        {
            // "/tr" : on propose les commandes qui commencent pareil.
            var close = Commands.Where(c => name.Length > 0 && c.Name.StartsWith(name, StringComparison.OrdinalIgnoreCase))
                .Select(c => "/" + c.Name).ToList();
            throw new UserException($"Commande inconnue : /{name}." +
                                    (close.Count > 0 ? $" Tu voulais dire {string.Join(" ou ", close)} ?" : " /help pour la liste."));
        }

        command.Execute(context, words[1..]);
    }
}
