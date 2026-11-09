#!/usr/bin/env bash
# prepare.sh <media dir> [docs dir]: sample data for the example service.
#
# Downloads photos from picsum.photos by stable id (Unsplash licence) into
# <dir>/photos, plus a smaller and a greyscale copy of a few, so that
# near-duplicates and their scores show up; then renders two slideshow clips
# into <dir>/clips with ffmpeg, four photos each, two seconds per photo, so an
# image query finds the clip and the second it appears at, and a clip finds
# its photos. A few photos also get copies into <dir>/copies, made with
# ffmpeg: re-encoded at a low quality, scaled to half, brightened, with a
# mark stamped on, as a PNG, and the file itself once more. To a perceptual
# hash those are the same picture, which is what the duplicates page shows.
# Files already there are kept: add your own, re-run freely.
#
# With a docs dir it also writes the sample documents of the text search
# page there, as Markdown files: the eval golden texts of this repository
# (zarf/eval/testdata), 84 short ones in English, Spanish and German under
# golden/ and 24 long ones with sections under long/. The service itself
# writes them (-write-docs), since it reads their YAML already.
set -euo pipefail

dir=${1:-data/media}
docs=${2:-}
photos="$dir/photos"
clips="$dir/clips"
copies="$dir/copies"

for tool in curl ffmpeg; do
	command -v "$tool" >/dev/null || { echo "prepare: $tool not found on PATH" >&2; exit 1; }
done
mkdir -p "$photos" "$clips" "$copies"

# picsum ids with a subject that reads at a glance
ids=(237 1025 1074 1084 1062 1069 1080 1011 1015 1016 1018 1020 1024 1035 1036 1039 1040 1043 1050 1057)
# the ones that also get a small and a greyscale copy
variants=(237 1025 1084)
# the ones that get near-exact copies for the duplicates page
duplicated=(1015 1074 1080)

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

edit() { # edit <in> <out> <ffmpeg video filter> [jpeg quality, 2 best to 31 worst]
	[ -s "$2" ] && return 0
	echo "  $2"
	ffmpeg -v error -y -i "$1" -vf "$3" -q:v "${4:-3}" "$2"
}

echo "copies into $copies"
for id in "${duplicated[@]}"; do
	src="$photos/picsum-$id.jpg"
	edit "$src" "$copies/picsum-$id-reencoded.jpg" "null" 18
	edit "$src" "$copies/picsum-$id-half.jpg" "scale=iw/2:ih/2"
	edit "$src" "$copies/picsum-$id-bright.jpg" "eq=brightness=0.08"
	edit "$src" "$copies/picsum-$id-marked.jpg" "drawbox=x=iw-110:y=ih-60:w=100:h=50:color=white:t=fill"
done
# the same pixels in another format, and the same bytes under another name
edit "$photos/picsum-1015.jpg" "$copies/picsum-1015-lossless.png" "null"
[ -s "$copies/picsum-1074-again.jpg" ] || { echo "  $copies/picsum-1074-again.jpg"; cp "$photos/picsum-1074.jpg" "$copies/picsum-1074-again.jpg"; }

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

echo "done: $(find "$photos" -type f | wc -l) photos, $(find "$copies" -type f | wc -l) copies, $(find "$clips" -type f | wc -l) clips in $dir"

if [ -n "$docs" ]; then
	echo "documents into $docs"
	go run . -write-docs "$docs"
fi
