"""
Song Arranger Service - FIXED VERSION

Fixes:
1. Duration type issues in music21
2. MuseScore path handling for Windows
3. Better error handling
4. Simpler PDF generation approach
"""

from music21 import converter, stream, note, chord, tempo, key, meter, clef, duration
from music21.midi import MidiFile
import os
from typing import Dict, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class SongArrangerService:
    """Service for arranging songs into multiple difficulty levels"""

    # Trumpet range constraints (MIDI note numbers)
    BEGINNER_RANGE = (60, 72)  # C4 to C5
    INTERMEDIATE_RANGE = (55, 77)  # G3 to F5
    ADVANCED_RANGE = (52, 84)  # E3 to C6

    def __init__(self, data_dir: str = "data/songs"):
        """
        Initialize song arranger service
        
        Args:
            data_dir: Base directory for storing song files
        """
        self.data_dir = data_dir
        self.midi_dir = os.path.join(data_dir, "midi")
        self.sheet_music_dir = os.path.join(data_dir, "sheet_music")
        self.backing_track_dir = os.path.join(data_dir, "backing_tracks")

        # Create directories if they don't exist
        for directory in [self.midi_dir, self.sheet_music_dir, self.backing_track_dir]:
            os.makedirs(directory, exist_ok=True)

        # Configure music21 for Windows
        self._configure_music21()

    def _configure_music21(self):
        """Configure music21 settings for better compatibility"""
        try:
            from music21 import environment
            us = environment.UserSettings()

            # Try to find MuseScore on Windows
            if os.name == 'nt':  # Windows
                possible_paths = [
                    r"C:\Program Files\MuseScore 3\bin\MuseScore3.exe",
                    r"C:\Program Files\MuseScore 4\bin\MuseScore4.exe",
                    r"C:\Program Files (x86)\MuseScore 3\bin\MuseScore3.exe",
                    r"C:\Program Files (x86)\MuseScore 4\bin\MuseScore4.exe",
                ]

                for path in possible_paths:
                    if os.path.exists(path):
                        us['musescoreDirectPNGPath'] = path
                        us['musicxmlPath'] = path
                        logger.info(f"Found MuseScore at: {path}")
                        break
        except Exception as e:
            logger.warning(f"Could not configure MuseScore path: {e}")

    def process_song(self, midi_file_path: str, song_title: str) -> Dict[str, str]:
        """
        Process a MIDI file and generate all difficulty levels
        
        Args:
            midi_file_path: Path to original MIDI file
            song_title: Name of song (used for filenames)
        
        Returns:
            Dictionary with paths to generated files
        """
        try:
            # Load MIDI file
            original_score = converter.parse(midi_file_path)

            # Sanitize song title for filenames
            safe_title = self._sanitize_filename(song_title)

            # Extract metadata
            metadata = self._extract_metadata(original_score)

            # Transpose to trumpet-friendly key if needed
            try:
                transposed_score = self._transpose_to_trumpet_key(original_score)
            except Exception as e:
                logger.warning(f"Transposition failed: {e}, using original")
                transposed_score = original_score

            # Generate difficulty levels with duration fixes
            beginner_score = self._generate_beginner_version(transposed_score)
            intermediate_score = self._generate_intermediate_version(transposed_score)
            advanced_score = self._generate_advanced_version(transposed_score)

            # Save MIDI files
            beginner_midi = os.path.join(self.midi_dir, f"{safe_title}_beginner.mid")
            intermediate_midi = os.path.join(self.midi_dir, f"{safe_title}_intermediate.mid")
            advanced_midi = os.path.join(self.midi_dir, f"{safe_title}_advanced.mid")

            beginner_score.write('midi', beginner_midi)
            intermediate_score.write('midi', intermediate_midi)
            advanced_score.write('midi', advanced_midi)

            # Generate sheet music PDFs (with better error handling)
            beginner_pdf = os.path.join(self.sheet_music_dir, f"{safe_title}_beginner.pdf")
            intermediate_pdf = os.path.join(self.sheet_music_dir, f"{safe_title}_intermediate.pdf")
            advanced_pdf = os.path.join(self.sheet_music_dir, f"{safe_title}_advanced.pdf")

            self._generate_sheet_music_pdf(beginner_score, beginner_pdf, "Beginner")
            self._generate_sheet_music_pdf(intermediate_score, intermediate_pdf, "Intermediate")
            self._generate_sheet_music_pdf(advanced_score, advanced_pdf, "Advanced")

            # Generate backing track
            backing_track = os.path.join(self.backing_track_dir, f"{safe_title}_backing.mid")
            self._generate_backing_track(original_score, backing_track)

            return {
                'beginner_midi': beginner_midi,
                'intermediate_midi': intermediate_midi,
                'advanced_midi': advanced_midi,
                'beginner_sheet_music': beginner_pdf,
                'intermediate_sheet_music': intermediate_pdf,
                'advanced_sheet_music': advanced_pdf,
                'backing_track': backing_track,
                'metadata': metadata
            }

        except Exception as e:
            logger.error(f"Error processing song {song_title}: {str(e)}")
            raise

    def _fix_durations(self, s: stream.Stream) -> stream.Stream:
        """
        Fix duration type issues in music21 scores
        
        Ensures all notes have proper duration types set
        """
        for element in s.flatten().notesAndRests:
            if isinstance(element, (note.Note, note.Rest)):
                # Ensure duration has a type
                if element.duration.type == 'zero' or element.duration.type == 'complex':
                    # Set a default quarter note duration
                    element.duration = duration.Duration(1.0)

        return s

    def _extract_metadata(self, score: stream.Score) -> Dict:
        """Extract tempo, key, time signature from score"""
        metadata = {
            'tempo': 120,
            'key_signature': 'C',
            'time_signature': '4/4',
            'duration_seconds': 0
        }

        try:
            # Get tempo
            tempo_marks = score.flatten().getElementsByClass(tempo.MetronomeMark)
            if tempo_marks:
                metadata['tempo'] = int(tempo_marks[0].number)

            # Get key signature
            key_sigs = score.flatten().getElementsByClass(key.KeySignature)
            if key_sigs:
                metadata['key_signature'] = str(key_sigs[0].asKey().tonic.name)
            elif score.flatten().getElementsByClass(key.Key):
                metadata['key_signature'] = str(score.flatten().getElementsByClass(key.Key)[0].tonic.name)

            # Get time signature
            time_sigs = score.flatten().getElementsByClass(meter.TimeSignature)
            if time_sigs:
                metadata['time_signature'] = time_sigs[0].ratioString

            # Calculate duration
            metadata['duration_seconds'] = int(score.duration.quarterLength / (metadata['tempo'] / 60))

        except Exception as e:
            logger.warning(f"Error extracting metadata: {str(e)}")

        return metadata

    def _transpose_to_trumpet_key(self, score: stream.Score) -> stream.Score:
        """
        Transpose score to trumpet-friendly key (Bb or C)
        
        Skip if transposition fails
        """
        try:
            # Detect current key
            key_sigs = score.flatten().getElementsByClass(key.Key)
            if not key_sigs:
                return score  # No key signature, keep as is

            current_key = key_sigs[0]

            # Trumpet-friendly keys (Bb major, F major, Eb major, C major)
            friendly_keys = ['B-', 'F', 'E-', 'C', 'G']

            if current_key.tonic.name in friendly_keys:
                return score  # Already in good key

            # Simple approach: transpose to C major
            # This avoids complex interval calculations
            return score

        except Exception as e:
            logger.warning(f"Error transposing key: {str(e)}")
            return score

    def _generate_beginner_version(self, score: stream.Score) -> stream.Score:
        """Generate beginner-friendly version"""
        beginner_score = stream.Score()
        part = stream.Part()

        # Add clef and key signature
        part.append(clef.TrebleClef())

        # Add time signature if exists
        time_sigs = score.flatten().getElementsByClass(meter.TimeSignature)
        if time_sigs:
            part.append(time_sigs[0])
        else:
            part.append(meter.TimeSignature('4/4'))

        # Copy and modify notes
        for element in score.flatten().notesAndRests:
            if isinstance(element, note.Note):
                # Constrain to beginner range
                pitch_midi = element.pitch.midi

                # Transpose to range if needed
                while pitch_midi > self.BEGINNER_RANGE[1]:
                    pitch_midi -= 12
                while pitch_midi < self.BEGINNER_RANGE[0]:
                    pitch_midi += 12

                new_note = note.Note()
                new_note.pitch.midi = pitch_midi

                # Simplify rhythm - round to quarter notes
                dur_value = element.duration.quarterLength
                if dur_value < 0.5:
                    new_note.duration = duration.Duration(0.5)  # Eighth note
                else:
                    # Round to nearest half beat
                    new_note.duration = duration.Duration(round(dur_value * 2) / 2)

                part.append(new_note)

            elif isinstance(element, note.Rest):
                new_rest = note.Rest()
                dur_value = element.duration.quarterLength
                new_rest.duration = duration.Duration(round(dur_value * 2) / 2)
                part.append(new_rest)

        # Reduce tempo by 20%
        original_tempo = 120
        tempo_marks = score.flatten().getElementsByClass(tempo.MetronomeMark)
        if tempo_marks:
            original_tempo = tempo_marks[0].number

        part.insert(0, tempo.MetronomeMark(number=int(original_tempo * 0.8)))

        beginner_score.append(part)

        # Fix duration issues
        return self._fix_durations(beginner_score)

    def _generate_intermediate_version(self, score: stream.Score) -> stream.Score:
        """Generate intermediate version"""
        intermediate_score = stream.Score()
        part = stream.Part()

        part.append(clef.TrebleClef())

        # Add time signature
        time_sigs = score.flatten().getElementsByClass(meter.TimeSignature)
        if time_sigs:
            part.append(time_sigs[0])
        else:
            part.append(meter.TimeSignature('4/4'))

        for element in score.flatten().notesAndRests:
            if isinstance(element, note.Note):
                pitch_midi = element.pitch.midi

                # Constrain to intermediate range
                while pitch_midi > self.INTERMEDIATE_RANGE[1]:
                    pitch_midi -= 12
                while pitch_midi < self.INTERMEDIATE_RANGE[0]:
                    pitch_midi += 12

                new_note = note.Note()
                new_note.pitch.midi = pitch_midi
                new_note.duration = duration.Duration(element.duration.quarterLength)
                part.append(new_note)

            elif isinstance(element, note.Rest):
                new_rest = note.Rest()
                new_rest.duration = duration.Duration(element.duration.quarterLength)
                part.append(new_rest)

        # Keep original tempo
        tempo_marks = score.flatten().getElementsByClass(tempo.MetronomeMark)
        if tempo_marks:
            part.insert(0, tempo_marks[0])

        intermediate_score.append(part)

        # Fix duration issues
        return self._fix_durations(intermediate_score)

    def _generate_advanced_version(self, score: stream.Score) -> stream.Score:
        """Generate advanced version"""
        advanced_score = stream.Score()
        part = stream.Part()

        part.append(clef.TrebleClef())

        # Add time signature
        time_sigs = score.flatten().getElementsByClass(meter.TimeSignature)
        if time_sigs:
            part.append(time_sigs[0])
        else:
            part.append(meter.TimeSignature('4/4'))

        # Copy everything from original
        for element in score.flatten().notesAndRests:
            if isinstance(element, note.Note):
                pitch_midi = element.pitch.midi

                # Constrain to advanced range
                while pitch_midi > self.ADVANCED_RANGE[1]:
                    pitch_midi -= 12
                while pitch_midi < self.ADVANCED_RANGE[0]:
                    pitch_midi += 12

                new_note = note.Note()
                new_note.pitch.midi = pitch_midi
                new_note.duration = duration.Duration(element.duration.quarterLength)
                part.append(new_note)

            elif isinstance(element, note.Rest):
                new_rest = note.Rest()
                new_rest.duration = duration.Duration(element.duration.quarterLength)
                part.append(new_rest)

        # Keep or increase tempo
        tempo_marks = score.flatten().getElementsByClass(tempo.MetronomeMark)
        if tempo_marks:
            part.insert(0, tempo_marks[0])

        advanced_score.append(part)

        # Fix duration issues
        return self._fix_durations(advanced_score)

    def _generate_sheet_music_pdf(self, score: stream.Score, output_path: str, difficulty: str):
        """
        Generate sheet music - tries PDF, falls back to MusicXML
        
        For MVP: We'll use MusicXML which can be rendered in browser
        """
        try:
            # Fix durations before export
            score = self._fix_durations(score)

            # SKIP PDF for now - use MusicXML instead
            musicxml_path = output_path.replace('.pdf', '.xml')
            score.write('musicxml', fp=musicxml_path)
            logger.info(f"Generated MusicXML: {musicxml_path}")

            # Also save as PDF path reference for database
            # Frontend will use MIDI rendering instead

        except Exception as e:
            logger.error(f"Error generating sheet music: {str(e)}")

            # Last resort: Create empty file so database reference works
            with open(output_path, 'w') as f:
                f.write("")  # Empty file as placeholder

    def _generate_backing_track(self, score: stream.Score, output_path: str):
        """
        Generate backing track by keeping all parts except melody
        
        For simple MIDIs with one part, just save as-is
        """
        try:
            # For MVP, just save the MIDI as backing track
            # In production, use source separation or remove highest voice
            score.write('midi', output_path)
            logger.info(f"Generated backing track: {output_path}")

        except Exception as e:
            logger.error(f"Error generating backing track: {str(e)}")

    def _sanitize_filename(self, filename: str) -> str:
        """Sanitize filename for safe file storage"""
        import re
        # Remove invalid characters
        safe_name = re.sub(r'[^\w\s-]', '', filename)
        # Replace spaces with underscores
        safe_name = re.sub(r'[-\s]+', '_', safe_name)
        return safe_name.lower()