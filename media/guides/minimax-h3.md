---
family: minimax-h3
kind: video
ref_token: "<Picture {n}>"
negative_prompt: false
max_refs: 9
---
No negatives.
Reference mode: write `motion_prompt` as these six labelled sections, in this order, each on its own line:
subject_definitions: who each reference is, and which reference wins if two conflict. Use {{ref:Name:slot}}.
summary: the clip in one sentence.
retention_analysis: what keeps the viewer watching.
detailed_description: the action in order, shot by shot, one camera move per shot.
overall_soundscape: voices, room tone and effects.
non_diegetic_music: music, or none.
Give each reference one job.
First frame mode: numbered shots, one camera move per shot. Describe sound as well, because this model always renders audio.
