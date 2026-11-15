using System.Collections;

namespace MiniChat.Data;

/// <summary>
/// Un tableau trop grand pour un tableau .NET (2,1 milliards d'éléments au plus) : des morceaux de ChunkSize
/// éléments mis bout à bout, lus avec un index en long. Le texte.txt de 5,6 Go fait déjà 1,6 milliard de tokens.
/// Tous les morceaux font ChunkSize éléments, sauf le dernier (juste ce qu'il faut : pas de place perdue).
/// </summary>
public sealed class ChunkedArray<T> : IEnumerable<T> where T : unmanaged
{
    /// <summary>268 millions d'éléments par morceau (2^28) : 512 Mo de tokens int16.</summary>
    public const int DefaultChunkSize = 1 << 28;

    private readonly T[][] _chunks;

    public ChunkedArray(T[][] chunks, int chunkSize)
    {
        if (chunks.Take(Math.Max(0, chunks.Length - 1)).Any(c => c.Length != chunkSize) || chunks is [.., { Length: 0 }]
            || chunks.Length > 0 && chunks[^1].Length > chunkSize)
            throw new ArgumentException("Chaque morceau fait chunkSize éléments, le dernier entre 1 et chunkSize.");
        _chunks = chunks;
        ChunkSize = chunkSize;
        Length = chunks.Sum(c => (long)c.Length);
    }

    /// <summary>Le tableau array, recopié en morceaux de chunkSize éléments.</summary>
    public static ChunkedArray<T> From(T[] array, int chunkSize = DefaultChunkSize)
    {
        var builder = new Builder(chunkSize);
        foreach (var value in array)
            builder.Add(value);
        return builder.Build();
    }

    public long Length { get; }
    public int ChunkSize { get; }

    /// <summary>Les morceaux, dans l'ordre (pour les écrire sur le disque).</summary>
    public IReadOnlyList<T[]> Chunks => _chunks;

    public T this[long index]
    {
        get => _chunks[index / ChunkSize][index % ChunkSize];
    }

    /// <summary>Copie destination.Length éléments à partir de start, qui peuvent être à cheval sur deux morceaux (ou plus).</summary>
    public void CopyTo(long start, Span<T> destination)
    {
        if (start < 0 || start + destination.Length > Length)
            throw new ArgumentOutOfRangeException(nameof(start));
        while (destination.Length > 0)
        {
            var chunk = _chunks[start / ChunkSize];
            int offset = (int)(start % ChunkSize);
            int count = Math.Min(destination.Length, chunk.Length - offset);
            chunk.AsSpan(offset, count).CopyTo(destination);
            destination = destination[count..];
            start += count;
        }
    }

    public IEnumerator<T> GetEnumerator()
    {
        foreach (var chunk in _chunks)
            foreach (var value in chunk)
                yield return value;
    }

    IEnumerator IEnumerable.GetEnumerator() => GetEnumerator();

    /// <summary>
    /// Remplit un ChunkedArray élément par élément, sans savoir à l'avance combien il y en aura : un morceau
    /// de plus quand le dernier est plein. Jamais de grand tableau recopié en plus grand ; seul le dernier
    /// morceau est raccourci à la fin.
    /// </summary>
    public sealed class Builder(int chunkSize = DefaultChunkSize)
    {
        private readonly List<T[]> _chunks = [];
        private T[]? _current;
        private int _used;

        public long Length => (long)_chunks.Count * chunkSize + _used;

        public void Add(T value)
        {
            if (_current is null || _used == _current.Length)
            {
                if (_current is not null)
                    _chunks.Add(_current);
                _current = new T[chunkSize];
                _used = 0;
            }
            _current[_used++] = value;
        }

        /// <summary>Un morceau entier d'un coup (lecture du cache) : chunk.Length vaut chunkSize, sauf pour le dernier.</summary>
        public void AddChunk(T[] chunk)
        {
            if (_current is not null)
                throw new InvalidOperationException("AddChunk après Add.");
            _chunks.Add(chunk);
        }

        public ChunkedArray<T> Build()
        {
            if (_current is not null && _used > 0)
            {
                if (_used < _current.Length)
                    Array.Resize(ref _current, _used);
                _chunks.Add(_current);
            }
            _current = null;
            return new ChunkedArray<T>([.. _chunks], chunkSize);
        }
    }
}
