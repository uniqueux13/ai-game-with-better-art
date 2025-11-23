import numpy as np
import wave

SAMPLE_RATE = 44100

# 1. Music Theory: Define exact Frequencies for notes
NOTES = {
    'rest': 0,
    'A2': 110.00, 'C3': 130.81, 'D3': 146.83, 'E3': 164.81, 'G3': 196.00,
    'A3': 220.00, 'C4': 261.63, 'D4': 293.66, 'E4': 329.63, 'G4': 392.00,
    'A4': 440.00, 'C5': 523.25, 'D5': 587.33, 'E5': 659.25, 'G5': 783.99,
}

def save_wav(filename, audio_data):
    # Normalize to prevent clipping
    max_val = np.max(np.abs(audio_data))
    if max_val > 0: audio_data = audio_data / max_val
    
    # Convert to 16-bit PCM
    audio_int16 = (audio_data * 32767).astype(np.int16)
    with wave.open(filename, 'w') as w:
        w.setnchannels(1) 
        w.setsampwidth(2) 
        w.setframerate(SAMPLE_RATE)
        w.writeframes(audio_int16.tobytes())
    print(f" -> Generated Melody: {filename}")

def synth_note(note_name, duration_sec, volume=0.5):
    """Creates a single musical note with a 'plucked' sound."""
    freq = NOTES.get(note_name, 0)
    t = np.linspace(0, duration_sec, int(SAMPLE_RATE * duration_sec), endpoint=False)
    
    if freq == 0:
        return np.zeros_like(t) # Silence for rests

    # RETRO SOUND: Square Wave (like NES/Gameboy)
    # np.sign(np.sin(...)) creates a blocky square wave
    raw_wave = np.sign(np.sin(2 * np.pi * freq * t)) * volume
    
    # ENVELOPE: Make it sound like a pluck (loud start, fade out)
    # This prevents the "continuous beep" sound
    decay = np.linspace(1.0, 0.2, len(t)) # Fade from 100% to 20%
    audio = raw_wave * decay
    
    # Quick anti-click fade at the very end
    fade_out = 500
    if len(audio) > fade_out:
        audio[-fade_out:] *= np.linspace(1, 0, fade_out)

    return audio

def compile_melody(filename, melody, bpm):
    """
    melody: List of tuples ('NOTE', beats)
    bpm: Beats per minute
    """
    beat_sec = 60.0 / bpm
    full_track = []
    
    # Repeat the melody 4 times to make the file longer
    for _ in range(4):
        for note, beats in melody:
            duration = beats * beat_sec
            # Leave a tiny gap between notes for articulation (staccato)
            note_audio = synth_note(note, duration * 0.90) 
            gap = np.zeros(int(SAMPLE_RATE * (duration * 0.10)))
            
            full_track.append(note_audio)
            full_track.append(gap)
            
    save_wav(filename, np.concatenate(full_track))

# --- DEFINE SONGS USING MUSIC THEORY ---

if __name__ == "__main__":
    print("Composing Retro Melodies...")

    # 1. MENU THEME: Slow, ominous A Minor Arpeggio
    # Pattern: Root, 5th, Octave, 5th
    menu_melody = [
        ('A2', 1), ('E3', 1), ('A3', 1), ('E3', 1),
        ('C3', 1), ('E3', 1), ('A3', 2),
    ]
    compile_melody("menu_theme.wav", menu_melody, bpm=90)

    # 2. LEVEL 1: C Minor Pentatonic "Walking" (The Chase Begins)
    # Typical "Spy/Action" bassline feel
    level1_melody = [
        ('C3', 0.5), ('C3', 0.5), ('E3', 0.5), ('G3', 0.5),
        ('A3', 0.5), ('G3', 0.5), ('E3', 0.5), ('C3', 0.5),
        ('D3', 0.5), ('D3', 0.5), ('F3', 0.5), ('A3', 0.5), # slight key shift
        ('C4', 1.0), ('rest', 1.0)
    ]
    compile_melody("level_1.wav", level1_melody, bpm=120)

    # 3. LEVEL 2: Faster, Higher Pitch (E Minor)
    # Urgent, repetitive alarms
    level2_melody = [
        ('E4', 0.25), ('G4', 0.25), ('E4', 0.25), ('B4', 0.25),
        ('E4', 0.25), ('G4', 0.25), ('E4', 0.25), ('D5', 0.25),
        ('E4', 0.25), ('G4', 0.25), ('E4', 0.25), ('B4', 0.25),
        ('A4', 0.5), ('G4', 0.5)
    ]
    compile_melody("level_2.wav", level2_melody, bpm=140)

    # 4. LEVEL 3: High Speed Panic
    # Very fast, erratic notes
    level3_melody = [
        ('A4', 0.25), ('C5', 0.25), ('E5', 0.25), ('A5', 0.25),
        ('G5', 0.25), ('E5', 0.25), ('C5', 0.25), ('G4', 0.25),
        ('A4', 0.25), ('rest', 0.25), ('A4', 0.25), ('rest', 0.25),
    ]
    compile_melody("level_3.wav", level3_melody, bpm=160)