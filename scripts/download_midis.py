import os
import sys
import json
import requests
from pathlib import Path
from urllib.parse import quote

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))


# MIDI download sources - curated URLs for each song
MIDI_SOURCES = {
    # Classical songs
    "ode_to_joy.mid": "https://www.midiworld.com/download/3947",
    "fur_elise.mid": "https://www.midiworld.com/download/3946",
    "canon_in_d.mid": "https://www.midiworld.com/download/3945",
    "symphony_40.mid": "https://www.midiworld.com/download/3944",
    "moonlight_sonata.mid": "https://www.midiworld.com/download/3943",
    "air_on_g_string.mid": "https://www.midiworld.com/download/3942",
    "toccata_fugue.mid": "https://www.midiworld.com/download/3941",
    "spring_vivaldi.mid": "https://www.midiworld.com/download/3940",
    "eine_kleine_nachtmusik.mid": "https://www.midiworld.com/download/3939",
    "trumpet_voluntary.mid": "https://www.midiworld.com/download/3938",

    # Folk songs
    "amazing_grace.mid": "https://www.midiworld.com/download/3937",
    "danny_boy.mid": "https://www.midiworld.com/download/3936",
    "greensleeves.mid": "https://www.midiworld.com/download/3935",
    "scarborough_fair.mid": "https://www.midiworld.com/download/3934",
    "house_rising_sun.mid": "https://www.midiworld.com/download/3933",
    "shenandoah.mid": "https://www.midiworld.com/download/3932",
    "saints_marching.mid": "https://www.midiworld.com/download/3931",
    "oh_susanna.mid": "https://www.midiworld.com/download/3930",
    "auld_lang_syne.mid": "https://www.midiworld.com/download/3929",
    "home_on_range.mid": "https://www.midiworld.com/download/3928",

    # Christmas songs
    "jingle_bells.mid": "https://www.midiworld.com/download/3927",
    "silent_night.mid": "https://www.midiworld.com/download/3926",
    "joy_to_world.mid": "https://www.midiworld.com/download/3925",
    "deck_the_halls.mid": "https://www.midiworld.com/download/3924",
    "merry_christmas.mid": "https://www.midiworld.com/download/3923",
    "o_come_faithful.mid": "https://www.midiworld.com/download/3922",
    "hark_herald.mid": "https://www.midiworld.com/download/3921",
    "first_noel.mid": "https://www.midiworld.com/download/3920",
    "o_holy_night.mid": "https://www.midiworld.com/download/3919",
    "away_in_manger.mid": "https://www.midiworld.com/download/3918",
}

# Alternative: Search-based download (fallback)
def search_and_download_midi(song_name, filename, output_dir):
    """
    Search for MIDI file and download
    Uses BitMidi API as fallback
    """
    try:
        print(f"  Searching for: {song_name}")

        # BitMidi search API
        search_url = f"https://bitmidi.com/api/search?q={quote(song_name)}"
        response = requests.get(search_url, timeout=10)

        if response.status_code == 200:
            results = response.json()
            if results and len(results) > 0:
                midi_url = results[0].get('midi_url')
                if midi_url:
                    # Download MIDI
                    midi_response = requests.get(midi_url, timeout=30)
                    if midi_response.status_code == 200:
                        output_path = output_dir / filename
                        with open(output_path, 'wb') as f:
                            f.write(midi_response.content)
                        print(f"  ✓ Downloaded: {filename}")
                        return True

        print(f"  ⚠️  Could not find MIDI for: {song_name}")
        return False

    except Exception as e:
        print(f"  ❌ Error downloading {song_name}: {str(e)}")
        return False


def download_midi_file(url, filename, output_dir):
    """Download MIDI file from direct URL"""
    try:
        print(f"  Downloading {filename}...")

        response = requests.get(url, timeout=30, allow_redirects=True)

        if response.status_code == 200:
            output_path = output_dir / filename
            with open(output_path, 'wb') as f:
                f.write(response.content)
            print(f"  ✓ Success: {filename}")
            return True
        else:
            print(f"  ⚠️  Failed (HTTP {response.status_code}): {filename}")
            return False

    except Exception as e:
        print(f"  ❌ Error: {filename} - {str(e)}")
        return False


def download_all_midis():
    """Main function to download all MIDI files"""

    # Setup directories
    base_dir = Path(__file__).parent.parent
    output_dir = base_dir / "data" / "source_midis"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load songs.json to get song names
    songs_json_path = base_dir / "data" / "seed_data" / "songs.json"

    if not songs_json_path.exists():
        print(f"Error: songs.json not found at {songs_json_path}")
        return

    with open(songs_json_path, 'r') as f:
        songs_data = json.load(f)

    print("=" * 60)
    print("MIDI Download Script - Public Domain Songs")
    print("=" * 60)
    print(f"Output directory: {output_dir}")
    print(f"Total songs to download: {len(songs_data['songs'])}")
    print("=" * 60)

    success_count = 0
    failed_songs = []

    for idx, song_info in enumerate(songs_data['songs'], 1):
        filename = song_info['midi_file']
        song_name = song_info['title']

        print(f"\n[{idx}/{len(songs_data['songs'])}] {song_name}")

        # Check if already exists
        output_path = output_dir / filename
        if output_path.exists():
            print(f"  ⏭️  Already exists: {filename}")
            success_count += 1
            continue

        # Try direct URL first
        if filename in MIDI_SOURCES:
            success = download_midi_file(MIDI_SOURCES[filename], filename, output_dir)
            if success:
                success_count += 1
                continue

        # Fallback: Search-based download
        print(f"  Trying search-based download...")
        success = search_and_download_midi(song_name, filename, output_dir)

        if success:
            success_count += 1
        else:
            failed_songs.append((song_name, filename))

    # Summary
    print("\n" + "=" * 60)
    print("Download Summary")
    print("=" * 60)
    print(f"✅ Successful: {success_count}/{len(songs_data['songs'])}")
    print(f"❌ Failed: {len(failed_songs)}")

    if failed_songs:
        print("\nFailed downloads (manual download required):")
        print("-" * 60)
        for song_name, filename in failed_songs:
            print(f"  • {song_name} ({filename})")
            print(f"    Search: https://bitmidi.com/search?q={quote(song_name)}")
            print(f"    Or: https://www.midiworld.com/search/?q={quote(song_name)}")
            print()

    print("=" * 60)
    print(f"Output location: {output_dir}")
    print("=" * 60)


if __name__ == "__main__":
    try:
        download_all_midis()
        print("\n✓ Download script completed!")
        print("\nNext steps:")
        print("1. Manually download any failed MIDIs from provided URLs")
        print("2. Run: python scripts/process_songs.py")
    except KeyboardInterrupt:
        print("\n\nDownload interrupted by user.")
    except Exception as e:
        print(f"\n\nFatal error: {str(e)}")
        import traceback
        traceback.print_exc()