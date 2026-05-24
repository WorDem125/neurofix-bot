import sys, subprocess, base64
from pathlib import Path

img = Path('/data/uploads/restore_demo_2.jpg').read_bytes()
print(f"Фото: {len(img)} байт")

work = Path('/tmp/dbg_input')
work.mkdir(exist_ok=True)
(work / 'test.jpg').write_bytes(img)

out = Path('/tmp/dbg_output')
out.mkdir(exist_ok=True)

cmd = [sys.executable, 'run.py',
       '--input_folder', str(work),
       '--output_folder', str(out),
       '--GPU', '0',
       '--with_scratch']

print('Запуск пайплайна...')
res = subprocess.run(cmd, cwd='/app/engines/old_photo',
                     capture_output=True, text=True, timeout=180)

print('returncode:', res.returncode)
print('--- STDOUT ---')
print(res.stdout[-3000:])
print('--- STDERR ---')
print(res.stderr[-3000:])
print('--- Файлы ---')
for f in out.rglob('*'):
    print(f)
