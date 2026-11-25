function youtubeId(url: string) {
  const m = url.match(/(?:youtu\.be\/|v=|embed\/|shorts\/)([\w-]{11})/);
  return m?.[1];
}

export default function Video({ url, day }: { url?: string; day: number }) {
  const id = url ? youtubeId(url) : undefined;
  if (!id) {
    return (
      <div className="video placeholder">
        <div>
          <div className="play">▶</div>
          <div>Video for Day {day} coming soon</div>
        </div>
      </div>
    );
  }
  return (
    <div className="video">
      <iframe
        src={`https://www.youtube-nocookie.com/embed/${id}`}
        title={`Day ${day} video`}
        allow="accelerometer; clipboard-write; encrypted-media; gyroscope; picture-in-picture"
        allowFullScreen
      />
    </div>
  );
}
