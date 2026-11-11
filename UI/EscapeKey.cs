namespace MiniChat.UI;

/// <summary>
/// Surveille la touche Échap pendant un travail long (réponse du modèle, entraînement) :
///   using var escape = EscapeKey.Watch();
///   travail(escape.Token);        // Token est annulé dès qu'on appuie sur Échap
/// Les autres touches tapées pendant ce temps sont ignorées. Entrée redirigée : rien à surveiller.
/// Une question posée pendant le travail (une confirmation) : using (escape.Pause()) { ... }.
/// </summary>
public sealed class EscapeKey : IDisposable
{
    private readonly CancellationTokenSource _escape = new();
    private readonly CancellationTokenSource _stop = new();
    private readonly Task? _watcher;

    // Pris par la boucle pendant qu'elle lit les touches, et par Pause tant que la question est posée.
    private readonly object _keys = new();

    private EscapeKey()
    {
        if (!Console.IsInputRedirected)
            _watcher = Task.Run(Loop);
    }

    public static EscapeKey Watch() => new();

    public CancellationToken Token => _escape.Token;

    /// <summary>Arrête de lire les touches jusqu'au Dispose du résultat : elles vont à la question posée.</summary>
    public IDisposable Pause()
    {
        Monitor.Enter(_keys);
        return new Resume(_keys);
    }

    // Ne lit une touche que si elle est déjà là : ReadKey ne bloque jamais, et le fil s'arrête dès Dispose.
    private void Loop()
    {
        while (!_stop.IsCancellationRequested)
        {
            lock (_keys)
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

    private sealed class Resume(object keys) : IDisposable
    {
        public void Dispose() => Monitor.Exit(keys);
    }
}
