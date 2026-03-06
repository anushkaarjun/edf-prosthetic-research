/**
 * Start backend (Flask on :5001) then frontend (Vite). One command for full app.
 */
const { spawn } = require('child_process');
const path = require('path');
const net = require('net');

const root = path.resolve(__dirname, '..');
const backendDir = path.join(root, 'backend');

function waitForPort(port, maxWaitMs = 180000) {
  return new Promise((resolve, reject) => {
    const start = Date.now();
    function tryConnect() {
      const sock = net.connect(port, '127.0.0.1', () => {
        sock.destroy();
        resolve();
      });
      sock.on('error', () => {
        if (Date.now() - start > maxWaitMs) return reject(new Error('Backend did not start in time'));
        setTimeout(tryConnect, 400);
      });
    }
    tryConnect();
  });
}

const backend = spawn('python3', ['app.py'], {
  cwd: backendDir,
  stdio: 'inherit',
  shell: false,
});

backend.on('error', (err) => {
  console.error('Failed to start backend:', err.message);
  console.error('Make sure you are in the project root and backend/app.py exists.');
  process.exit(1);
});

console.log('Starting backend on http://localhost:5001 ...');
console.log('(Loading data + model may take 1–2 min. Please wait.)');
waitForPort(5001)
  .then(() => {
    console.log('Backend is up. Starting frontend...');
    const vite = spawn('npx', ['vite'], { cwd: root, stdio: 'inherit', shell: true });
    vite.on('exit', (code) => {
      backend.kill();
      process.exit(code ?? 0);
    });
    process.on('SIGINT', () => {
      backend.kill();
      vite.kill();
      process.exit(0);
    });
    process.on('SIGTERM', () => {
      backend.kill();
      vite.kill();
      process.exit(0);
    });
  })
  .catch((err) => {
    console.error(err.message);
    backend.kill();
    process.exit(1);
  });
