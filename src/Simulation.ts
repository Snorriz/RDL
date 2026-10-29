import * as THREE from 'three';
import { DQNAgent } from './ReinforcementLearning';

export interface SimStats {
  episode: number;
  epsilon: number;
  avgReward: number;
  lastReward: number;
  distance: number;
  memorySize: number;
  step: number;
  catchCount: number;
  ballSpeed: number;
  lastAction: number;
}

export class Simulation {
  scene: THREE.Scene;
  camera: THREE.PerspectiveCamera;
  renderer: THREE.WebGLRenderer;
  ball: THREE.Mesh;
  ballOuter: THREE.Mesh;
  cube: THREE.Mesh;
  cubeWireframe: THREE.LineSegments;
  ground: THREE.Mesh;
  trail: THREE.Points;
  trailPositions: Float32Array;
  trailColors: Float32Array;
  trailIndex: number;
  agent: DQNAgent;
  ballVelocity: THREE.Vector3;
  arenaSize: number;
  maxSteps: number;
  currentStep: number;
  isRunning: boolean;
  speed: number;
  onStatsUpdate: ((stats: SimStats) => void) | null;
  animationId: number | null;
  particles: THREE.Points;
  particlePositions: Float32Array;
  particleVelocities: number[][];
  particleLifetimes: number[];
  ambientParticles: THREE.Points;
  ambientParticlePositions: Float32Array;
  rewardFlash: number;
  container: HTMLElement;
  ringEffect: THREE.Mesh;
  ringScale: number;
  ringOpacity: number;
  connectionLine: THREE.Line;
  time: number;
  catchCount: number;

  // Physics constants
  readonly FRICTION = 0.985;        // velocity damping per step
  readonly IMPULSE_STRENGTH = 0.35;  // push magnitude
  readonly MAX_VELOCITY = 2.5;       // velocity cap
  readonly WALL_RESTITUTION = 0.6;   // bounce energy retention
  readonly BALL_RADIUS = 0.45;
  readonly BALL_START_X = 0;
  readonly BALL_START_Z = 0;

  // Visual indicators
  impulseArrow: THREE.ArrowHelper | null;
  impulseArrowTimer: number;
  velocityArrow: THREE.ArrowHelper | null;

  constructor(container: HTMLElement) {
    this.container = container;
    this.time = 0;
    this.catchCount = 0;
    this.impulseArrow = null;
    this.impulseArrowTimer = 0;
    this.velocityArrow = null;

    // Scene
    this.scene = new THREE.Scene();

    // Camera
    this.camera = new THREE.PerspectiveCamera(
      55,
      container.clientWidth / container.clientHeight,
      0.1,
      1000
    );
    this.camera.position.set(0, 22, 20);
    this.camera.lookAt(0, 0, 0);

    // Renderer
    this.renderer = new THREE.WebGLRenderer({
      antialias: true,
      alpha: true,
      powerPreference: 'high-performance',
    });
    this.renderer.setSize(container.clientWidth, container.clientHeight);
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    this.renderer.shadowMap.enabled = true;
    this.renderer.shadowMap.type = THREE.PCFSoftShadowMap;
    this.renderer.toneMapping = THREE.ACESFilmicToneMapping;
    this.renderer.toneMappingExposure = 1.2;
    container.appendChild(this.renderer.domElement);

    // Arena
    this.arenaSize = 14;

    // Background gradient sphere
    const bgGeo = new THREE.SphereGeometry(80, 32, 32);
    const bgMat = new THREE.ShaderMaterial({
      side: THREE.BackSide,
      uniforms: {
        colorTop: { value: new THREE.Color(0x0a0020) },
        colorBottom: { value: new THREE.Color(0x000510) },
        colorAccent: { value: new THREE.Color(0x0a0030) },
      },
      vertexShader: `
        varying vec3 vWorldPosition;
        void main() {
          vec4 worldPos = modelMatrix * vec4(position, 1.0);
          vWorldPosition = worldPos.xyz;
          gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
        }
      `,
      fragmentShader: `
        uniform vec3 colorTop;
        uniform vec3 colorBottom;
        uniform vec3 colorAccent;
        varying vec3 vWorldPosition;
        void main() {
          float h = normalize(vWorldPosition).y;
          vec3 col = mix(colorBottom, colorTop, max(h, 0.0));
          col = mix(col, colorAccent, smoothstep(-0.2, 0.2, h) * 0.5);
          gl_FragColor = vec4(col, 1.0);
        }
      `,
    });
    const bgMesh = new THREE.Mesh(bgGeo, bgMat);
    this.scene.add(bgMesh);

    // Lights
    const ambientLight = new THREE.AmbientLight(0x202040, 0.4);
    this.scene.add(ambientLight);

    const mainLight = new THREE.DirectionalLight(0xeeeeff, 0.6);
    mainLight.position.set(15, 25, 10);
    mainLight.castShadow = true;
    mainLight.shadow.mapSize.width = 2048;
    mainLight.shadow.mapSize.height = 2048;
    mainLight.shadow.camera.near = 0.5;
    mainLight.shadow.camera.far = 60;
    mainLight.shadow.camera.left = -20;
    mainLight.shadow.camera.right = 20;
    mainLight.shadow.camera.top = 20;
    mainLight.shadow.camera.bottom = -20;
    mainLight.shadow.bias = -0.001;
    this.scene.add(mainLight);

    const rimLight = new THREE.DirectionalLight(0x4444ff, 0.3);
    rimLight.position.set(-10, 10, -15);
    this.scene.add(rimLight);

    // Ground with custom shader (neon grid)
    const groundGeo = new THREE.PlaneGeometry(this.arenaSize * 2.5, this.arenaSize * 2.5, 1, 1);
    const groundMat = new THREE.ShaderMaterial({
      uniforms: {
        time: { value: 0 },
        arenaSize: { value: this.arenaSize },
        ballPos: { value: new THREE.Vector2(0, 0) },
        cubePos: { value: new THREE.Vector2(0, 0) },
      },
      vertexShader: `
        varying vec2 vUv;
        varying vec3 vPos;
        void main() {
          vUv = uv;
          vPos = position;
          gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
        }
      `,
      fragmentShader: `
        uniform float time;
        uniform float arenaSize;
        uniform vec2 ballPos;
        uniform vec2 cubePos;
        varying vec2 vUv;
        varying vec3 vPos;
        
        void main() {
          vec2 pos = vPos.xz;
          
          // Grid
          vec2 grid = abs(fract(pos / 2.0 - 0.5) - 0.5) / (fwidth(pos / 2.0) * 1.5);
          float line = min(grid.x, grid.y);
          float gridAlpha = 1.0 - min(line, 1.0);
          
          // Arena boundary glow
          float distToEdge = min(
            min(abs(pos.x + arenaSize), abs(pos.x - arenaSize)),
            min(abs(pos.y + arenaSize), abs(pos.y - arenaSize))
          );
          float edgeGlow = smoothstep(2.0, 0.0, distToEdge) * 0.6;
          
          // Ball proximity glow
          float distToBall = length(pos - ballPos);
          float ballGlow = smoothstep(5.0, 0.0, distToBall) * 0.15;
          
          // Cube proximity glow
          float distToCube = length(pos - cubePos);
          float cubeGlow = smoothstep(5.0, 0.0, distToCube) * 0.15;
          
          // Base color
          vec3 baseColor = vec3(0.02, 0.02, 0.05);
          vec3 gridColor = vec3(0.1, 0.15, 0.4) * gridAlpha * 0.4;
          vec3 edgeColor = vec3(0.2, 0.1, 0.8) * edgeGlow;
          vec3 ballColor = vec3(0.0, 1.0, 0.5) * ballGlow;
          vec3 cubeColor = vec3(1.0, 0.2, 0.2) * cubeGlow;
          
          vec3 finalColor = baseColor + gridColor + edgeColor + ballColor + cubeColor;
          
          // Fade at edges
          float fade = smoothstep(arenaSize * 1.2, arenaSize * 0.8, max(abs(pos.x), abs(pos.y)));
          
          gl_FragColor = vec4(finalColor * fade, 1.0);
        }
      `,
    });
    this.ground = new THREE.Mesh(groundGeo, groundMat);
    this.ground.rotation.x = -Math.PI / 2;
    this.ground.position.y = -0.01;
    this.scene.add(this.ground);

    // Shadow receiving plane
    const shadowGeo = new THREE.PlaneGeometry(this.arenaSize * 2.5, this.arenaSize * 2.5);
    const shadowMat = new THREE.ShadowMaterial({ opacity: 0.3 });
    const shadowPlane = new THREE.Mesh(shadowGeo, shadowMat);
    shadowPlane.rotation.x = -Math.PI / 2;
    shadowPlane.position.y = 0;
    shadowPlane.receiveShadow = true;
    this.scene.add(shadowPlane);

    // Arena border (neon)
    const borderPoints = [
      new THREE.Vector3(-this.arenaSize, 0, -this.arenaSize),
      new THREE.Vector3(this.arenaSize, 0, -this.arenaSize),
      new THREE.Vector3(this.arenaSize, 0, this.arenaSize),
      new THREE.Vector3(-this.arenaSize, 0, this.arenaSize),
      new THREE.Vector3(-this.arenaSize, 0, -this.arenaSize),
    ];
    const borderGeo = new THREE.BufferGeometry().setFromPoints(borderPoints);
    const borderMat = new THREE.LineBasicMaterial({ color: 0x4444ff, linewidth: 2 });
    const border = new THREE.Line(borderGeo, borderMat);
    border.position.y = 0.05;
    this.scene.add(border);

    // Corner pillars
    const pillarGeo = new THREE.CylinderGeometry(0.08, 0.08, 3, 8);
    const pillarMat = new THREE.MeshStandardMaterial({
      color: 0x4444ff,
      emissive: 0x2222ff,
      emissiveIntensity: 0.5,
    });
    const corners = [
      [-this.arenaSize, -this.arenaSize],
      [this.arenaSize, -this.arenaSize],
      [this.arenaSize, this.arenaSize],
      [-this.arenaSize, this.arenaSize],
    ];
    corners.forEach(([x, z]) => {
      const pillar = new THREE.Mesh(pillarGeo, pillarMat);
      pillar.position.set(x, 1.5, z);
      this.scene.add(pillar);
      
      const pLight = new THREE.PointLight(0x4444ff, 0.5, 5);
      pLight.position.set(x, 3, z);
      this.scene.add(pLight);
    });

    // Ball (Agent) - inner sphere
    const ballGeo = new THREE.SphereGeometry(this.BALL_RADIUS, 64, 64);
    const ballMat = new THREE.MeshStandardMaterial({
      color: 0x00ff88,
      roughness: 0.1,
      metalness: 0.9,
      emissive: 0x00ff88,
      emissiveIntensity: 0.8,
    });
    this.ball = new THREE.Mesh(ballGeo, ballMat);
    this.ball.position.set(this.BALL_START_X, this.BALL_RADIUS + 0.1, this.BALL_START_Z);
    this.ball.castShadow = true;
    this.scene.add(this.ball);

    // Ball outer glow sphere
    const ballOuterGeo = new THREE.SphereGeometry(0.65, 32, 32);
    const ballOuterMat = new THREE.MeshBasicMaterial({
      color: 0x00ff88,
      transparent: true,
      opacity: 0.15,
      side: THREE.BackSide,
    });
    this.ballOuter = new THREE.Mesh(ballOuterGeo, ballOuterMat);
    this.ball.add(this.ballOuter);

    // Ball point light
    const ballLight = new THREE.PointLight(0x00ff88, 3, 8);
    ballLight.position.set(0, 0, 0);
    this.ball.add(ballLight);

    // Cube (Target) - solid
    const cubeGeo = new THREE.BoxGeometry(0.9, 0.9, 0.9);
    const cubeMat = new THREE.MeshStandardMaterial({
      color: 0xff3344,
      roughness: 0.1,
      metalness: 0.9,
      emissive: 0xff2233,
      emissiveIntensity: 0.6,
    });
    this.cube = new THREE.Mesh(cubeGeo, cubeMat);
    this.cube.position.set(6, 0.6, 6);
    this.cube.castShadow = true;
    this.scene.add(this.cube);

    // Cube wireframe
    const cubeEdges = new THREE.EdgesGeometry(new THREE.BoxGeometry(1.1, 1.1, 1.1));
    const cubeWireMat = new THREE.LineBasicMaterial({ color: 0xff6666, transparent: true, opacity: 0.6 });
    this.cubeWireframe = new THREE.LineSegments(cubeEdges, cubeWireMat);
    this.cube.add(this.cubeWireframe);

    // Cube light
    const cubeLight = new THREE.PointLight(0xff3344, 3, 8);
    this.cube.add(cubeLight);

    // Ring effect (for catch animation)
    const ringGeo = new THREE.RingGeometry(0.5, 0.7, 64);
    const ringMat = new THREE.MeshBasicMaterial({
      color: 0xffaa00,
      transparent: true,
      opacity: 0,
      side: THREE.DoubleSide,
    });
    this.ringEffect = new THREE.Mesh(ringGeo, ringMat);
    this.ringEffect.rotation.x = -Math.PI / 2;
    this.ringEffect.position.y = 0.1;
    this.ringScale = 0;
    this.ringOpacity = 0;
    this.scene.add(this.ringEffect);

    // Connection line (ball to cube)
    const connGeo = new THREE.BufferGeometry();
    connGeo.setAttribute('position', new THREE.Float32BufferAttribute([0, 0, 0, 0, 0, 0], 3));
    const connMat = new THREE.LineBasicMaterial({
      color: 0x4488ff,
      transparent: true,
      opacity: 0.2,
    });
    this.connectionLine = new THREE.Line(connGeo, connMat);
    this.scene.add(this.connectionLine);

    // Trail particles
    const trailCount = 300;
    this.trailPositions = new Float32Array(trailCount * 3);
    this.trailColors = new Float32Array(trailCount * 3);
    this.trailIndex = 0;
    for (let i = 0; i < trailCount; i++) {
      this.trailPositions[i * 3 + 1] = -10;
      this.trailColors[i * 3] = 0;
      this.trailColors[i * 3 + 1] = 1;
      this.trailColors[i * 3 + 2] = 0.5;
    }
    const trailGeo = new THREE.BufferGeometry();
    trailGeo.setAttribute('position', new THREE.BufferAttribute(this.trailPositions, 3));
    trailGeo.setAttribute('color', new THREE.BufferAttribute(this.trailColors, 3));
    const trailMat = new THREE.PointsMaterial({
      size: 0.12,
      vertexColors: true,
      transparent: true,
      opacity: 0.8,
      blending: THREE.AdditiveBlending,
      depthWrite: false,
    });
    this.trail = new THREE.Points(trailGeo, trailMat);
    this.scene.add(this.trail);

    // Catch particles
    const particleCount = 150;
    this.particlePositions = new Float32Array(particleCount * 3);
    this.particleVelocities = [];
    this.particleLifetimes = [];
    for (let i = 0; i < particleCount; i++) {
      this.particlePositions[i * 3 + 1] = -10;
      this.particleVelocities.push([0, 0, 0]);
      this.particleLifetimes.push(0);
    }
    const particleGeo = new THREE.BufferGeometry();
    particleGeo.setAttribute('position', new THREE.BufferAttribute(this.particlePositions, 3));
    const particleMat = new THREE.PointsMaterial({
      color: 0xffaa44,
      size: 0.18,
      transparent: true,
      opacity: 0.9,
      blending: THREE.AdditiveBlending,
      depthWrite: false,
    });
    this.particles = new THREE.Points(particleGeo, particleMat);
    this.scene.add(this.particles);

    // Ambient floating particles
    const ambientCount = 300;
    this.ambientParticlePositions = new Float32Array(ambientCount * 3);
    for (let i = 0; i < ambientCount; i++) {
      this.ambientParticlePositions[i * 3] = (Math.random() - 0.5) * 40;
      this.ambientParticlePositions[i * 3 + 1] = Math.random() * 15;
      this.ambientParticlePositions[i * 3 + 2] = (Math.random() - 0.5) * 40;
    }
    const ambientGeo = new THREE.BufferGeometry();
    ambientGeo.setAttribute('position', new THREE.BufferAttribute(this.ambientParticlePositions, 3));
    const ambientMat = new THREE.PointsMaterial({
      color: 0x6644ff,
      size: 0.06,
      transparent: true,
      opacity: 0.4,
      blending: THREE.AdditiveBlending,
      depthWrite: false,
    });
    this.ambientParticles = new THREE.Points(ambientGeo, ambientMat);
    this.scene.add(this.ambientParticles);

    // Agent: state = [ballX, ballZ, ballVx, ballVz, targetX, targetZ, distToTarget, speed]
    // Actions: 0=no push, 1-8=push in 8 directions
    this.agent = new DQNAgent(8, 9, [64, 32]);
    this.ballVelocity = new THREE.Vector3(0, 0, 0);
    this.maxSteps = 300;
    this.currentStep = 0;
    this.isRunning = false;
    this.speed = 1;
    this.onStatsUpdate = null;
    this.animationId = null;
    this.rewardFlash = 0;

    // Handle resize
    const resizeHandler = () => {
      this.camera.aspect = this.container.clientWidth / this.container.clientHeight;
      this.camera.updateProjectionMatrix();
      this.renderer.setSize(this.container.clientWidth, this.container.clientHeight);
    };
    window.addEventListener('resize', resizeHandler);
  }

  getState(): number[] {
    const dist = this.getDistance();
    const speed = Math.sqrt(this.ballVelocity.x ** 2 + this.ballVelocity.z ** 2);
    return [
      this.ball.position.x / this.arenaSize,
      this.ball.position.z / this.arenaSize,
      this.ballVelocity.x / this.MAX_VELOCITY,
      this.ballVelocity.z / this.MAX_VELOCITY,
      this.cube.position.x / this.arenaSize,
      this.cube.position.z / this.arenaSize,
      dist / (this.arenaSize * 1.414),
      speed / this.MAX_VELOCITY,
    ];
  }

  getDistance(): number {
    const dx = this.ball.position.x - this.cube.position.x;
    const dz = this.ball.position.z - this.cube.position.z;
    return Math.sqrt(dx * dx + dz * dz);
  }

  applyImpulse(action: number) {
    if (action === 0) return; // No push

    // 8 directions: 1=N, 2=NE, 3=E, 4=SE, 5=S, 6=SW, 7=W, 8=NW
    const angle = ((action - 1) / 8) * Math.PI * 2;
    const impulseX = Math.cos(angle) * this.IMPULSE_STRENGTH;
    const impulseZ = Math.sin(angle) * this.IMPULSE_STRENGTH;

    // Apply impulse to velocity
    this.ballVelocity.x += impulseX;
    this.ballVelocity.z += impulseZ;

    // Cap velocity
    const speed = Math.sqrt(this.ballVelocity.x ** 2 + this.ballVelocity.z ** 2);
    if (speed > this.MAX_VELOCITY) {
      this.ballVelocity.x = (this.ballVelocity.x / speed) * this.MAX_VELOCITY;
      this.ballVelocity.z = (this.ballVelocity.z / speed) * this.MAX_VELOCITY;
    }

    // Show impulse arrow visually
    this.showImpulseArrow(impulseX, impulseZ);
  }

  showImpulseArrow(dx: number, dz: number) {
    if (this.impulseArrow) {
      this.scene.remove(this.impulseArrow);
    }
    const dir = new THREE.Vector3(dx, 0, dz).normalize();
    const length = 2.0;
    this.impulseArrow = new THREE.ArrowHelper(dir, new THREE.Vector3(this.ball.position.x, 0.5, this.ball.position.z), length, 0xffff00, 0.4, 0.3);
    this.scene.add(this.impulseArrow);
    this.impulseArrowTimer = 15; // frames to show
  }

  updatePhysics() {
    // Apply friction
    this.ballVelocity.x *= this.FRICTION;
    this.ballVelocity.z *= this.FRICTION;

    // Stop very slow velocities
    if (Math.abs(this.ballVelocity.x) < 0.001) this.ballVelocity.x = 0;
    if (Math.abs(this.ballVelocity.z) < 0.001) this.ballVelocity.z = 0;

    // Update position
    this.ball.position.x += this.ballVelocity.x;
    this.ball.position.z += this.ballVelocity.z;

    // Wall collisions with bounce
    const limit = this.arenaSize - this.BALL_RADIUS;
    
    if (this.ball.position.x > limit) {
      this.ball.position.x = limit;
      this.ballVelocity.x = -this.ballVelocity.x * this.WALL_RESTITUTION;
    } else if (this.ball.position.x < -limit) {
      this.ball.position.x = -limit;
      this.ballVelocity.x = -this.ballVelocity.x * this.WALL_RESTITUTION;
    }

    if (this.ball.position.z > limit) {
      this.ball.position.z = limit;
      this.ballVelocity.z = -this.ballVelocity.z * this.WALL_RESTITUTION;
    } else if (this.ball.position.z < -limit) {
      this.ball.position.z = -limit;
      this.ballVelocity.z = -this.ballVelocity.z * this.WALL_RESTITUTION;
    }

    // Ball rotation based on velocity (visual rolling effect)
    const speed = Math.sqrt(this.ballVelocity.x ** 2 + this.ballVelocity.z ** 2);
    if (speed > 0.001) {
      const rotationAxis = new THREE.Vector3(-this.ballVelocity.z, 0, this.ballVelocity.x).normalize();
      const rotationAngle = speed / this.BALL_RADIUS;
      this.ball.rotateOnWorldAxis(rotationAxis, rotationAngle);
    }
  }

  moveCube() {
    const oldPos = this.cube.position.clone();
    
    // Move cube to new random position
    let newX: number, newZ: number;
    do {
      newX = (Math.random() - 0.5) * this.arenaSize * 1.6;
      newZ = (Math.random() - 0.5) * this.arenaSize * 1.6;
    } while (Math.sqrt((newX - this.BALL_START_X) ** 2 + (newZ - this.BALL_START_Z) ** 2) < 5);
    
    this.cube.position.x = newX;
    this.cube.position.z = newZ;

    // Spawn particles at old position
    this.spawnParticles(oldPos.x, oldPos.y, oldPos.z);
    
    // Trigger ring effect
    this.ringEffect.position.x = oldPos.x;
    this.ringEffect.position.z = oldPos.z;
    this.ringScale = 0.5;
    this.ringOpacity = 1.0;
    
    this.rewardFlash = 1.0;
    this.catchCount++;
  }

  spawnParticles(x: number, y: number, z: number) {
    for (let i = 0; i < this.particleVelocities.length; i++) {
      this.particlePositions[i * 3] = x;
      this.particlePositions[i * 3 + 1] = y + 0.5;
      this.particlePositions[i * 3 + 2] = z;
      const angle = Math.random() * Math.PI * 2;
      const spd = Math.random() * 0.2 + 0.05;
      this.particleVelocities[i] = [
        Math.cos(angle) * spd,
        Math.random() * 0.15 + 0.05,
        Math.sin(angle) * spd,
      ];
      this.particleLifetimes[i] = 1.0;
    }
  }

  updateParticles() {
    for (let i = 0; i < this.particleVelocities.length; i++) {
      if (this.particleLifetimes[i] > 0) {
        this.particlePositions[i * 3] += this.particleVelocities[i][0];
        this.particlePositions[i * 3 + 1] += this.particleVelocities[i][1];
        this.particlePositions[i * 3 + 2] += this.particleVelocities[i][2];
        this.particleVelocities[i][1] -= 0.003;
        this.particleLifetimes[i] -= 0.015;
        
        if (this.particleLifetimes[i] <= 0) {
          this.particlePositions[i * 3 + 1] = -10;
        }
      }
    }
    this.particles.geometry.attributes.position.needsUpdate = true;
  }

  updateTrail() {
    const idx = this.trailIndex % (this.trailPositions.length / 3);
    this.trailPositions[idx * 3] = this.ball.position.x;
    this.trailPositions[idx * 3 + 1] = 0.15;
    this.trailPositions[idx * 3 + 2] = this.ball.position.z;
    
    // Color based on speed
    const speed = Math.sqrt(this.ballVelocity.x ** 2 + this.ballVelocity.z ** 2);
    const t = Math.min(speed / this.MAX_VELOCITY, 1);
    this.trailColors[idx * 3] = t;           // R: more red at high speed
    this.trailColors[idx * 3 + 1] = 1 - t * 0.5; // G
    this.trailColors[idx * 3 + 2] = 0.5 * (1 - t); // B
    
    this.trailIndex++;
    this.trail.geometry.attributes.position.needsUpdate = true;
    this.trail.geometry.attributes.color.needsUpdate = true;
  }

  updateAmbientParticles() {
    for (let i = 0; i < this.ambientParticlePositions.length / 3; i++) {
      this.ambientParticlePositions[i * 3 + 1] += Math.sin(this.time * 0.5 + i) * 0.002;
      this.ambientParticlePositions[i * 3] += Math.cos(this.time * 0.3 + i * 0.7) * 0.001;
    }
    this.ambientParticles.geometry.attributes.position.needsUpdate = true;
  }

  step(): number {
    const state = this.getState();
    const action = this.agent.act(state);
    const prevDist = this.getDistance();

    // Apply impulse (push) - this is the agent's action
    this.applyImpulse(action);

    // Run physics simulation
    this.updatePhysics();

    // Calculate reward
    const newDist = this.getDistance();
    let reward = 0;
    let done = false;

    // Reward for getting closer to target
    reward += (prevDist - newDist) * 15;

    // Big reward for catching target
    if (newDist < 1.2) {
      reward += 100;
      done = true;
      this.moveCube();
    }

    // Small penalty for hitting walls
    const limit = this.arenaSize - this.BALL_RADIUS;
    if (
      Math.abs(this.ball.position.x) >= limit - 0.1 ||
      Math.abs(this.ball.position.z) >= limit - 0.1
    ) {
      reward -= 2;
    }

    // Small time penalty
    reward -= 0.05;

    // Bonus for maintaining momentum toward target
    const speed = Math.sqrt(this.ballVelocity.x ** 2 + this.ballVelocity.z ** 2);
    if (speed > 0.1) {
      const toTargetX = (this.cube.position.x - this.ball.position.x) / newDist;
      const toTargetZ = (this.cube.position.z - this.ball.position.z) / newDist;
      const velDirX = this.ballVelocity.x / speed;
      const velDirZ = this.ballVelocity.z / speed;
      const alignment = toTargetX * velDirX + toTargetZ * velDirZ;
      reward += alignment * 0.5 * speed;
    }

    this.currentStep++;
    if (this.currentStep >= this.maxSteps) {
      done = true;
    }

    const nextState = this.getState();
    this.agent.remember(state, action, reward, nextState, done);
    this.agent.replay();

    this.updateTrail();

    if (done) {
      this.agent.endEpisode();
      this.currentStep = 0;
      // Reset ball to start position (always same point!)
      this.ball.position.set(this.BALL_START_X, this.BALL_RADIUS + 0.1, this.BALL_START_Z);
      this.ballVelocity.set(0, 0, 0);
      // Reset ball rotation
      this.ball.rotation.set(0, 0, 0);
    }

    return reward;
  }

  animate = () => {
    if (!this.isRunning) return;

    this.time += 0.016;

    const stepsPerFrame = this.speed;
    for (let i = 0; i < stepsPerFrame; i++) {
      this.step();
    }

    // Visual updates
    this.updateParticles();
    this.updateAmbientParticles();

    // Animate cube
    this.cube.rotation.y += 0.025;
    this.cube.rotation.x = Math.sin(this.time * 2) * 0.15;
    this.cube.position.y = 0.6 + Math.sin(this.time * 3) * 0.15;
    this.cubeWireframe.rotation.y -= 0.01;
    this.cubeWireframe.rotation.z += 0.005;

    // Animate ball glow
    const bScale = 1 + Math.sin(this.time * 4) * 0.03;
    this.ballOuter.scale.set(
      1 + Math.sin(this.time * 2) * 0.1,
      1 + Math.sin(this.time * 2) * 0.1,
      1 + Math.sin(this.time * 2) * 0.1
    );
    (this.ballOuter.material as THREE.MeshBasicMaterial).opacity = 0.1 + Math.sin(this.time * 3) * 0.05;

    // Ball emissive based on speed
    const speed = Math.sqrt(this.ballVelocity.x ** 2 + this.ballVelocity.z ** 2);
    const speedRatio = Math.min(speed / this.MAX_VELOCITY, 1);
    const ballMat = this.ball.material as THREE.MeshStandardMaterial;
    ballMat.emissiveIntensity = 0.5 + speedRatio * 1.5;

    // Ring effect animation
    if (this.ringOpacity > 0) {
      this.ringScale += 0.15;
      this.ringOpacity -= 0.025;
      this.ringEffect.scale.set(this.ringScale, this.ringScale, 1);
      (this.ringEffect.material as THREE.MeshBasicMaterial).opacity = Math.max(0, this.ringOpacity);
    }

    // Reward flash
    if (this.rewardFlash > 0) {
      this.rewardFlash -= 0.02;
      const mat = this.cube.material as THREE.MeshStandardMaterial;
      mat.emissiveIntensity = 0.6 + this.rewardFlash * 3;
    }

    // Impulse arrow fade
    if (this.impulseArrowTimer > 0) {
      this.impulseArrowTimer--;
      if (this.impulseArrowTimer <= 0 && this.impulseArrow) {
        this.scene.remove(this.impulseArrow);
        this.impulseArrow = null;
      }
    }

    // Connection line
    const connPos = this.connectionLine.geometry.attributes.position as THREE.BufferAttribute;
    connPos.setXYZ(0, this.ball.position.x, 0.3, this.ball.position.z);
    connPos.setXYZ(1, this.cube.position.x, 0.3, this.cube.position.z);
    connPos.needsUpdate = true;
    const dist = this.getDistance();
    (this.connectionLine.material as THREE.LineBasicMaterial).opacity = Math.max(0.05, 0.3 - dist * 0.02);

    // Update ground shader uniforms
    const groundUniforms = (this.ground.material as THREE.ShaderMaterial).uniforms;
    groundUniforms.time.value = this.time;
    groundUniforms.ballPos.value.set(this.ball.position.x, this.ball.position.z);
    groundUniforms.cubePos.value.set(this.cube.position.x, this.cube.position.z);

    // Camera smooth follow
    const targetCamX = this.ball.position.x * 0.2;
    const targetCamZ = this.ball.position.z * 0.2 + 20;
    this.camera.position.x += (targetCamX - this.camera.position.x) * 0.015;
    this.camera.position.z += (targetCamZ - this.camera.position.z) * 0.015;
    this.camera.position.y = 22 + Math.sin(this.time * 0.5) * 0.5;
    this.camera.lookAt(
      this.ball.position.x * 0.1,
      0,
      this.ball.position.z * 0.1
    );

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
        catchCount: this.catchCount,
        ballSpeed: speed,
        lastAction: this.agent.lastAction,
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
    this.agent = new DQNAgent(8, 9, [64, 32]);
    this.ball.position.set(this.BALL_START_X, this.BALL_RADIUS + 0.1, this.BALL_START_Z);
    this.ball.rotation.set(0, 0, 0);
    this.cube.position.set(
      (Math.random() - 0.5) * this.arenaSize * 1.5,
      0.6,
      (Math.random() - 0.5) * this.arenaSize * 1.5
    );
    this.ballVelocity.set(0, 0, 0);
    this.currentStep = 0;
    this.catchCount = 0;
    this.trailIndex = 0;
    for (let i = 0; i < this.trailPositions.length / 3; i++) {
      this.trailPositions[i * 3 + 1] = -10;
    }
    this.trail.geometry.attributes.position.needsUpdate = true;
    if (this.impulseArrow) {
      this.scene.remove(this.impulseArrow);
      this.impulseArrow = null;
    }
  }

  dispose() {
    this.stop();
    this.renderer.dispose();
    if (this.renderer.domElement.parentElement) {
      this.renderer.domElement.parentElement.removeChild(this.renderer.domElement);
    }
  }
}
