from sqlalchemy.orm import Session
from app.database.connection import SessionLocal
from app.database import crud

def seed_exercises():
    """Seed database with initial exercises"""
    db: Session = SessionLocal()

    exercises = [
        # Breathing Exercises
        {
            "title": "Basic Breath Control",
            "description": "Learn to take deep, consistent breaths",
            "technique": "breathing",
            "difficulty": "beginner",
            "instructions": """1. Stand or sit with good posture
2. Breathe in through your nose for 4 counts
3. Hold for 2 counts
4. Breathe out through your mouth for 6 counts
5. Repeat 10 times
6. Now try with the trumpet - play a long tone on middle C
7. Focus on steady, controlled air flow""",
            "duration_minutes": 5,
            "order_index": 1
        },
        {
            "title": "Breath Support Exercise",
            "description": "Develop strong breath support from the diaphragm",
            "technique": "breathing",
            "difficulty": "intermediate",
            "instructions": """1. Place your hand on your stomach
2. Breathe deeply - your hand should move out
3. Play a crescendo on a single note (start soft, get loud)
4. Focus on pushing air from your diaphragm
5. Do this on: G, C, E, G (low to high)
6. Each note for 8 counts""",
            "duration_minutes": 10,
            "order_index": 2
        },

        # Tone Quality Exercises
        {
            "title": "Long Tones - Foundation",
            "description": "Build a beautiful, consistent tone",
            "technique": "tone",
            "difficulty": "beginner",
            "instructions": """1. Play middle C
2. Hold the note for 10 seconds
3. Focus on: steady pitch, consistent volume, clear tone
4. Repeat on: D, E, F, G
5. Rest between each note
6. Listen carefully to your tone quality""",
            "duration_minutes": 8,
            "order_index": 3
        },
        {
            "title": "Tone Quality Development",
            "description": "Refine your tone with varied dynamics",
            "technique": "tone",
            "difficulty": "intermediate",
            "instructions": """1. Play G (middle of staff)
2. Start soft (p), crescendo to loud (f), decrescendo back to soft
3. Keep pitch and tone quality consistent throughout
4. Repeat on: F, E, D, C
5. Focus on not letting tone get harsh when loud
6. Don't let tone get breathy when soft""",
            "duration_minutes": 12,
            "order_index": 4
        },

        # Rhythm Exercises
        {
            "title": "Steady Beat Foundation",
            "description": "Develop rock-solid rhythm",
            "technique": "rhythm",
            "difficulty": "beginner",
            "instructions": """1. Set metronome to 80 BPM
2. Clap quarter notes with the metronome for 16 beats
3. Now play quarter notes on middle C
4. Focus on hitting exactly with the click
5. Try half notes (2 beats each)
6. Try whole notes (4 beats each)
7. Keep the beat steady and consistent""",
            "duration_minutes": 10,
            "order_index": 5
        },
    ]

    try:
        for ex_data in exercises:
            exercise = crud.create_exercise(db, **ex_data)
            print(f"✓ Created exercise: {exercise.title}")

        print(f"\n✓ Successfully seeded {len(exercises)} exercises")

    except Exception as e:
        print(f"✗ Error seeding exercises: {e}")
        db.rollback()
    finally:
        db.close()

if __name__ == "__main__":
    seed_exercises()