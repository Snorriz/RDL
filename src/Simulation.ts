import * as THREE from 'three';
import { DQNAgent } from './ReinforcementLearning';

export class Simulation {
  scene: THREE.Scene;
  camera: THREE.PerspectiveCamera;
  renderer: THREE.WebGLRenderer;
  ball: THREE.Mesh;
  cube: THREE.Mesh;
  ground: THREE.Mesh;
  trail: THREE.Line;
  trailPoints: THREE.Vector3[];
  agent: DQNAgent;
  ballVelocity: THREE.Vector3;
  arenaSize: number;
  maxSteps: number;
  currentStep: number;
  isRunning: boolean;
  speed: number;
  onStatsUpdate: ((stats: SimStats) => void) | null;
  animationId: number | null;
  cubeGlow: THREE.PointLight;
  ballGlow: THREE.PointLight;
  particles: THREE.Points;
  particlePositions: Float32Array;
  particleVelocities: number[][];
  rewardFlash: number;
  gridHelper: THREE.GridHelper;

  constructor(container: HTMLElement) {
    // Scene setup
    this.scene = new THREE.Scene();
    this.scene.background = new THREE.Color(0x0a0a1a);
    this.scene.fog = new THREE.Fog(0x0a0a1a, 30, 60);

    // Camera
    this.camera = new THREE.PerspectiveCamera(
      60,
      container.clientWidth / container.clientHeight,
      0.1,
      1000
    );
    this.camera.position.set(0, 18, 18);
    this.camera.lookAt(0, 0, 0);

    // Renderer
    this.renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
    this.renderer.setSize(container.clientWidth, container.clientHeight);
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    this.renderer.shadowMap.enabled = true;
    this.renderer.shadowMap.type = THREE.PCFSoftShadowMap;
    container.appendChild(this.renderer.domElement);

    // Arena
    this.arenaSize = 12;

    // Lights
    const ambientLight = new THREE.AmbientLight(0x404060, 0.5);
    this.scene.add(ambientLight);

    const dirLight = new THREE.DirectionalLight(0xffffff, 0.8);
    dirLight.position.set(10, 20, 10);
    dirLight.castShadow = true;
    dirLight.shadow.mapSize.width = 2048;
    dirLight.shadow.mapSize.height = 2048;
    dirLight.shadow.camera.near = 0.5;
    dirLight.shadow.camera.far = 50;
    dirLight.shadow.camera.left = -15;
    dirLight.shadow.camera.right = 15;
    dirLight.shadow.camera.top = 15;
    dirLight.shadow.camera.bottom = -15;
    this.scene.add(dirLight);

    // Grid
    this.gridHelper = new THREE.GridHelper(this.arenaSize * 2, 24, 0x1a1a3a, 0x1a1a3a);
    this.gridHelper.position.y = 0.01;
    this.scene.add(this.gridHelper);

    // Ground
    const groundGeo = new THREE.PlaneGeometry(this.arenaSize * 2, this.arenaSize * 2);
    const groundMat = new THREE.MeshStandardMaterial({
      color: 0x111122,
      roughness: 0.8,
      metalness: 0.2,
    });
    this.ground = new THREE.Mesh(groundGeo, groundMat);
    this.ground.rotation.x = -Math.PI / 2;
    this.ground.receiveShadow = true;
    this.scene.add(this.ground);

    // Arena border
    const borderGeo = new THREE.BoxGeometry(this.arenaSize * 2, 0.3, this.arenaSize * 2);
    const edges = new THREE.EdgesGeometry(borderGeo);
    const borderLine = new THREE.LineSegments(
      edges,
      new THREE.LineBasicMaterial({ color: 0x3333ff, linewidth: 2 })
    );
    borderLine.position.y = 0.15;
    this.scene.add(borderLine);

    // Ball (Agent)
    const ballGeo = new THREE.SphereGeometry(0.5, 32, 32);
    const ballMat = new THREE.MeshStandardMaterial({
      color: 0x00ff88,
      roughness: 0.3,
      metalness: 0.7,
      emissive: 0x00ff88,
      emissiveIntensity: 0.3,
    });
    this.ball = new THREE.Mesh(ballGeo, ballMat);
    this.ball.position.set(0, 0.5, 0);
    this.ball.castShadow = true;
    this.scene.add(this.ball);

    // Ball glow
    this.ballGlow = new THREE.PointLight(0x00ff88, 2, 5);
    this.ball.add(this.ballGlow);

    // Cube (Target)
    const cubeGeo = new THREE.BoxGeometry(1, 1, 1);
    const cubeMat = new THREE.MeshStandardMaterial({
      color: 0xff4444,
      roughness: 0.3,
      metalness: 0.7,
      emissive: 0xff4444,
      emissiveIntensity: 0.3,
    });
    this.cube = new THREE.Mesh(cubeGeo, cubeMat);
    this.cube.position.set(
      (Math.random() - 0.5) * this.arenaSize * 1.5,
      0.5,
      (Math.random() - 0.5) * this.arenaSize * 1.5
    );
    this.cube.castShadow = true;
    this.scene.add(this.cube);

    // Cube glow
    this.cubeGlow = new THREE.PointLight(0xff4444, 2, 5);
    this.cube.add(this.cubeGlow);

    // Trail
    this.trailPoints = [];
    const trailGeo = new THREE.BufferGeometry();
    const trailMat = new THREE.LineBasicMaterial({
      color: 0x00ff88,
      transparent: true,
      opacity: 0.5,
    });
    this.trail = new THREE.Line(trailGeo, trailMat);
    this.scene.add(this.trail);

    // Particles
    const particleCount = 100;
    this.particlePositions = new Float32Array(particleCount * 3);
    this.particleVelocities = [];
    for (let i = 0; i < particleCount; i++) {
      this.particlePositions[i * 3] = 0;
      this.particlePositions[i * 3 + 1] = -10;
      this.particlePositions[i * 3 + 2] = 0;
      this.particleVelocities.push([0, 0, 0]);
    }
    const particleGeo = new THREE.BufferGeometry();
    particleGeo.setAttribute('position', new THREE.BufferAttribute(this.particlePositions, 3));
    const particleMat = new THREE.PointsMaterial({
      color: 0xffff00,
      size: 0.15,
      transparent: true,
      opacity: 0.8,
    });
    this.particles = new THREE.Points(particleGeo, particleMat);
    this.scene.add(this.particles);

    // Agent
    // State: [ballX, ballZ, cubeX, cubeZ, velX, velZ]
    // Actions: 0-7 (8 directions)
    this.agent = new DQNAgent(6, 8, [128, 64]);
    this.ballVelocity = new THREE.Vector3(0, 0, 0);
    this.maxSteps = 200;
    this.currentStep = 0;
    this.isRunning = false;
    this.speed = 1;
    this.onStatsUpdate = null;
    this.animationId = null;
    this.rewardFlash = 0;

    // Handle resize
    window.addEventListener('resize', () => {
      this.camera.aspect = container.clientWidth / container.clientHeight;
      this.camera.updateProjectionMatrix();
      this.renderer.setSize(container.clientWidth, container.clientHeight);
    });
  }

  getState(): number[] {
    return [
      this.ball.position.x / this.arenaSize,
      this.ball.position.z / this.arenaSize,
      this.cube.position.x / this.arenaSize,
      this.cube.position.z / this.arenaSize,
      this.ballVelocity.x,
      this.ballVelocity.z,
    ];
  }

  getDistance(): number {
    const dx = this.ball.position.x - this.cube.position.x;
    const dz = this.ball.position.z - this.cube.position.z;
    return Math.sqrt(dx * dx + dz * dz);
  }

  moveCube() {
    const oldPos = this.cube.position.clone();
    
    // Move cube to a new random position
    this.cube.position.x = (Math.random() - 0.5) * this.arenaSize * 1.6;
    this.cube.position.z = (Math.random() - 0.5) * this.arenaSize * 1.6;

    // Spawn particles at old position
    this.spawnParticles(oldPos.x, oldPos.y, oldPos.z);
    this.rewardFlash = 1.0;
  }

  spawnParticles(x: number, y: number, z: number) {
    for (let i = 0; i < this.particleVelocities.length; i++) {
      this.particlePositions[i * 3] = x;
      this.particlePositions[i * 3 + 1] = y;
      this.particlePositions[i * 3 + 2] = z;
      this.particleVelocities[i] = [
        (Math.random() - 0.5) * 0.3,
        Math.random() * 0.2,
        (Math.random() - 0.5) * 0.3,
      ];
    }
  }

  updateParticles() {
    for (let i = 0; i < this.particleVelocities.length; i++) {
      this.particlePositions[i * 3] += this.particleVelocities[i][0];
      this.particlePositions[i * 3 + 1] += this.particleVelocities[i][1];
      this.particlePositions[i * 3 + 2] += this.particleVelocities[i][2];
      this.particleVelocities[i][1] -= 0.005; // gravity
    }
    this.particles.geometry.attributes.position.needsUpdate = true;
  }

  step(): number {
    const state = this.getState();
    const action = this.agent.act(state);
    const prevDistance = this.getDistance();

    // Convert action to direction (8 directions)
    const angle = (action / 8) * Math.PI * 2;
    const moveSpeed = 0.15;
    this.ballVelocity.x = Math.cos(angle) * moveSpeed;
    this.ballVelocity.z = Math.sin(angle) * moveSpeed;

    // Move ball
    this.ball.position.x += this.ballVelocity.x;
    this.ball.position.z += this.ballVelocity.z;

    // Clamp to arena
    this.ball.position.x = Math.max(-this.arenaSize, Math.min(this.arenaSize, this.ball.position.x));
    this.ball.position.z = Math.max(-this.arenaSize, Math.min(this.arenaSize, this.ball.position.z));

    // Calculate reward
    const newDistance = this.getDistance();
    let reward = 0;
    let done = false;

    // Reward for getting closer
    reward += (prevDistance - newDistance) * 10;

    // Check if reached cube
    if (newDistance < 1.2) {
      reward += 100;
      done = true;
      this.moveCube();
    }

    // Penalty for hitting walls
    if (
      Math.abs(this.ball.position.x) >= this.arenaSize ||
      Math.abs(this.ball.position.z) >= this.arenaSize
    ) {
      reward -= 1;
    }

    // Small step penalty
    reward -= 0.1;

    this.currentStep++;
    if (this.currentStep >= this.maxSteps) {
      done = true;
    }

    // Store experience
    const nextState = this.getState();
    this.agent.remember(state, action, reward, nextState, done);
    this.agent.replay();

    // Update trail
    this.trailPoints.push(new THREE.Vector3(this.ball.position.x, 0.1, this.ball.position.z));
    if (this.trailPoints.length > 100) {
      this.trailPoints.shift();
    }
    const trailGeo = new THREE.BufferGeometry().setFromPoints(this.trailPoints);
    this.trail.geometry.dispose();
    this.trail.geometry = trailGeo;

    if (done) {
      this.agent.endEpisode();
      this.currentStep = 0;
      this.trailPoints = [];
      // Reset ball position
      this.ball.position.x = (Math.random() - 0.5) * this.arenaSize;
      this.ball.position.z = (Math.random() - 0.5) * this.arenaSize;
      this.ballVelocity.set(0, 0, 0);
    }

    return reward;
  }

  animate = () => {
    if (!this.isRunning) return;

    // Run multiple steps per frame for faster training
    const stepsPerFrame = this.speed;
    for (let i = 0; i < stepsPerFrame; i++) {
      this.step();
    }

    // Visual updates
    this.updateParticles();

    // Animate cube rotation
    this.cube.rotation.y += 0.02;
    this.cube.rotation.x += 0.01;

    // Animate ball pulsing
    const scale = 1 + Math.sin(Date.now() * 0.005) * 0.05;
    this.ball.scale.set(scale, scale, scale);

    // Reward flash
    if (this.rewardFlash > 0) {
      this.rewardFlash -= 0.02;
      const mat = this.cube.material as THREE.MeshStandardMaterial;
      mat.emissiveIntensity = 0.3 + this.rewardFlash * 2;
      this.cubeGlow.intensity = 2 + this.rewardFlash * 5;
    }

    // Camera follows ball slightly
    const targetCamX = this.ball.position.x * 0.3;
    const targetCamZ = this.ball.position.z * 0.3 + 18;
    this.camera.position.x += (targetCamX - this.camera.position.x) * 0.02;
    this.camera.position.z += (targetCamZ - this.camera.position.z) * 0.02;
    this.camera.lookAt(0, 0, 0);

    // Update stats
    if (this.onStatsUpdate) {
      this.onStatsUpdate({
        episode: this.agent.episodeCount,
        epsilon: this.agent.epsilon,
        avgReward: this.agent.getAverageReward(20),
        lastReward: this.agent.totalReward,
        distance: this.getDistance(),
        memorySize: this.agent.memory.length,
        step: this.currentStep,
      });
    }

    this.renderer.render(this.scene, this.camera);
    this.animationId = requestAnimationFrame(this.animate);
  };

  start() {
    this.isRunning = true;
    this.animate();
  }

  stop() {
    this.isRunning = false;
    if (this.animationId) {
      cancelAnimationFrame(this.animationId);
    }
  }

  setSpeed(speed: number) {
    this.speed = speed;
  }

  reset() {
    this.agent = new DQNAgent(6, 8, [128, 64]);
    this.ball.position.set(0, 0.5, 0);
    this.cube.position.set(
      (Math.random() - 0.5) * this.arenaSize * 1.5,
      0.5,
      (Math.random() - 0.5) * this.arenaSize * 1.5
    );
    this.currentStep = 0;
    this.trailPoints = [];
  }

  dispose() {
    this.stop();
    this.renderer.dispose();
  }
}

export interface SimStats {
  episode: number;
  epsilon: number;
  avgReward: number;
  lastReward: number;
  distance: number;
  memorySize: number;
  step: number;
}
