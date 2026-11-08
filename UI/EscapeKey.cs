namespace MiniChat.UI;

/// <summary>
/// Surveille la touche Échap pendant un travail long (réponse du modèle, entraînement) :
///   using var escape = EscapeKey.Watch();
///   travail(escape.Token);        // Token est annulé dès qu'on appuie sur Échap
/// Les autres touches tapées pendant ce temps sont ignorées. Entrée redirigée : rien à surveiller.
/// </summary>
public sealed class EscapeKey : IDisposable
{
    private readonly CancellationTokenSource _escape = new();
    private readonly CancellationTokenSource _stop = new();
    private readonly Task? _watcher;

    private EscapeKey()
    {
        if (!Console.IsInputRedirected)
            _watcher = Task.Run(Loop);
    }

    public static EscapeKey Watch() => new();

    public CancellationToken Token => _escape.Token;

    // Ne lit une touche que si elle est déjà là : ReadKey ne bloque jamais, et le fil s'arrête dès Dispose.
    private void Loop()
    {
        while (!_stop.IsCancellationRequested)
        {
            while (Console.KeyAvailable)
                if (Console.ReadKey(intercept: true).Key == ConsoleKey.Escape)
                    _escape.Cancel();
            Thread.Sleep(50);
        }
    }

    /// <summary>Arrête la surveillance avant de rendre la main : la saisie suivante reçoit toutes ses touches.</summary>
    public void Dispose()
    {
        _stop.Cancel();
        _watcher?.Wait();
        _stop.Dispose();
        _escape.Dispose();
    }
}
