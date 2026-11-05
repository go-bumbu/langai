#!/usr/bin/env bash
# prepare.sh <dir>: sample media for the example service.
#
# Downloads photos from picsum.photos by stable id (Unsplash licence) into
# <dir>/photos, plus a smaller and a greyscale copy of a few, so that
# near-duplicates and their scores show up; then renders two slideshow clips
# into <dir>/clips with ffmpeg, four photos each, two seconds per photo, so an
# image query finds the clip and the second it appears at, and a clip finds
# its photos. Files already there are kept: add your own, re-run freely.
set -euo pipefail

dir=${1:-data/media}
photos="$dir/photos"
clips="$dir/clips"

for tool in curl ffmpeg; do
	command -v "$tool" >/dev/null || { echo "prepare: $tool not found on PATH" >&2; exit 1; }
done
mkdir -p "$photos" "$clips"

# picsum ids with a subject that reads at a glance
ids=(237 1025 1074 1084 1062 1069 1080 1011 1015 1016 1018 1020 1024 1035 1036 1039 1040 1043 1050 1057)
# the ones that also get a small and a greyscale copy
variants=(237 1025 1084)

fetch() { # fetch <url> <file>
	[ -s "$2" ] && return 0
	echo "  $2"
	curl -fsSL --retry 3 -o "$2.part" "$1" && mv "$2.part" "$2"
}

echo "photos into $photos"
for id in "${ids[@]}"; do
	fetch "https://picsum.photos/id/$id/800/600.jpg" "$photos/picsum-$id.jpg"
done
for id in "${variants[@]}"; do
	fetch "https://picsum.photos/id/$id/400/300.jpg" "$photos/picsum-$id-small.jpg"
	fetch "https://picsum.photos/id/$id/800/600.jpg?grayscale" "$photos/picsum-$id-gray.jpg"
done

slideshow() { # slideshow <out> <photo>...
	local out=$1; shift
	[ -s "$out" ] && return 0
	echo "  $out"
	local args=()
	for p in "$@"; do args+=(-loop 1 -t 2 -i "$p"); done
	local n=$#
	local filter=""
	for ((i = 0; i < n; i++)); do filter+="[$i:v]scale=640:480:force_original_aspect_ratio=decrease,pad=640:480:-1:-1,setsar=1,fps=25[v$i];"; done
	for ((i = 0; i < n; i++)); do filter+="[v$i]"; done
	filter+="concat=n=$n:v=1:a=0[v]"
	# the native mpeg4 encoder: no libx264 needed; faststart puts the index first, so the
	# file plays from a pipe and the browser starts it sooner
	ffmpeg -v error -y "${args[@]}" -filter_complex "$filter" -map "[v]" -c:v mpeg4 -q:v 3 -pix_fmt yuv420p -movflags +faststart "$out"
}

echo "clips into $clips"
slideshow "$clips/slideshow-1.mp4" "$photos/picsum-237.jpg" "$photos/picsum-1074.jpg" "$photos/picsum-1011.jpg" "$photos/picsum-1040.jpg"
slideshow "$clips/slideshow-2.mp4" "$photos/picsum-1025-gray.jpg" "$photos/picsum-1062.jpg" "$photos/picsum-1084.jpg" "$photos/picsum-1050.jpg"

echo "done: $(find "$photos" -type f | wc -l) photos, $(find "$clips" -type f | wc -l) clips in $dir"
