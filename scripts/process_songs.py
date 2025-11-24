"""
Process Public Domain Songs Script

This script:
1. Reads song metadata from songs.json
2. Processes each MIDI file through SongArrangerService
3. Generates 3 difficulty levels
4. Creates backing tracks
5. Generates sheet music PDFs
6. Seeds database with song entries

Usage:
    python scripts/process_songs.py
"""

import os
import sys
import json
from pathlib import Path

# Add parent directory to path to import app modules
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.services.song_arranger_service import SongArrangerService
from app.database.connection import SessionLocal
from app.database import crud


def process_all_songs():
    """Process all songs from songs.json"""

    # Load song metadata
    songs_json_path = Path(__file__).parent.parent / "data" / "seed_data" / "songs.json"

    if not songs_json_path.exists():
        print(f"Error: songs.json not found at {songs_json_path}")
        return

    with open(songs_json_path, 'r') as f:
        songs_data = json.load(f)

    # Initialize arranger service
    arranger = SongArrangerService()

    # Database session
    db = SessionLocal()

    print(f"Processing {len(songs_data['songs'])} songs...")
    print("=" * 60)

    success_count = 0
    error_count = 0

    for idx, song_data in enumerate(songs_data['songs'], 1):
        try:
            print(f"\n[{idx}/{len(songs_data['songs'])}] Processing: {song_data['title']}")

            # Get MIDI file path
            midi_filename = song_data['midi_file']
            midi_path = Path(__file__).parent.parent / "data" / "source_midis" / midi_filename

            if not midi_path.exists():
                print(f"  ⚠️  Warning: MIDI file not found: {midi_path}")
                error_count += 1
                continue

            # Process song
            print(f"  🎵 Processing MIDI file...")
            result = arranger.process_song(str(midi_path), song_data['title'])

            print(f"  ✓ Generated 3 difficulty levels")
            print(f"  ✓ Generated sheet music PDFs")
            print(f"  ✓ Generated backing track")

            # Create database entry
            print(f"  💾 Saving to database...")
            db_song = crud.create_song(
                db=db,
                title=song_data['title'],
                composer=song_data.get('composer'),
                artist=song_data.get('artist'),
                genre=song_data['genre'],
                tempo=result['metadata'].get('tempo'),
                key_signature=result['metadata'].get('key_signature'),
                time_signature=result['metadata'].get('time_signature'),
                duration_seconds=result['metadata'].get('duration_seconds'),
                beginner_midi_path=result['beginner_midi'],
                intermediate_midi_path=result['intermediate_midi'],
                advanced_midi_path=result['advanced_midi'],
                beginner_sheet_music_path=result['beginner_sheet_music'],
                intermediate_sheet_music_path=result['intermediate_sheet_music'],
                advanced_sheet_music_path=result['advanced_sheet_music'],
                backing_track_path=result['backing_track'],
                is_public_domain=song_data.get('is_public_domain', True),
                order_index=idx
            )

            print(f"  ✅ Success! Song ID: {db_song.id}")
            success_count += 1

        except Exception as e:
            print(f"  ❌ Error processing {song_data['title']}: {str(e)}")
            error_count += 1
            continue

    db.close()

    print("\n" + "=" * 60)
    print(f"Processing complete!")
    print(f"✅ Successful: {success_count}")
    print(f"❌ Errors: {error_count}")
    print("=" * 60)


if __name__ == "__main__":
    try:
        process_all_songs()
    except KeyboardInterrupt:
        print("\n\nProcessing interrupted by user.")
    except Exception as e:
        print(f"\n\nFatal error: {str(e)}")
        import traceback
        traceback.print_exc()