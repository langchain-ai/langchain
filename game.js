// Grow Some Mushrooms - Game Logic
class MushroomGame {
    constructor() {
        this.canvas = document.getElementById('gameCanvas');
        this.ctx = this.canvas.getContext('2d');
        this.width = this.canvas.width;
        this.height = this.canvas.height;
        
        // Game state
        this.level = 1;
        this.score = 0;
        this.spores = 10;
        this.robux = 50; // Premium currency
        this.gameState = 'playing'; // 'playing', 'aiming', 'launching', 'levelComplete'
        
        // Robux system
        this.robuxOfferShown = false;
        this.robuxOfferAvailable = true;
        this.lastOfferLevel = 0;
        this.currentOffer = null;
        
        // Slingshot mechanics
        this.slingshot = { x: 100, y: this.height - 100 };
        this.aiming = false;
        this.aimAngle = 0;
        this.aimPower = 0;
        this.mousePos = { x: 0, y: 0 };
        
        // Projectiles (spores)
        this.projectiles = [];
        
        // Game objects
        this.soilPatches = [];
        this.mushrooms = [];
        this.obstacles = [];
        this.waterPatches = [];
        
        // Visual effects
        this.particles = [];
        
        // Colors (Atari palette)
        this.colors = {
            background: '#001122',
            soil: '#8B4513',
            mushroom: '#FF6B35',
            mushroomSpot: '#FFFFFF',
            rock: '#696969',
            water: '#4169E1',
            spore: '#00FF00',
            slingshot: '#FFAA00',
            trajectory: '#00FF0040'
        };
        
        this.init();
        this.setupEventListeners();
        this.gameLoop();
    }
    
    init() {
        this.generateLevel(this.level);
        this.updateUI();
    }
    
    generateLevel(levelNum) {
        this.soilPatches = [];
        this.mushrooms = [];
        this.obstacles = [];
        this.waterPatches = [];
        this.projectiles = [];
        this.particles = [];
        
        // Generate soil patches (targets to grow mushrooms on)
        const soilCount = Math.min(3 + levelNum, 12);
        for (let i = 0; i < soilCount; i++) {
            const x = 200 + Math.random() * (this.width - 400);
            const y = 100 + Math.random() * (this.height - 200);
            const radius = 20 + Math.random() * 15;
            
            this.soilPatches.push({
                x: x,
                y: y,
                radius: radius,
                hasMushroom: false,
                fertility: 1.0
            });
        }
        
        // Generate obstacles (rocks)
        const rockCount = Math.floor(levelNum / 2) + 1;
        for (let i = 0; i < rockCount; i++) {
            const x = 200 + Math.random() * (this.width - 400);
            const y = 50 + Math.random() * (this.height - 150);
            const radius = 15 + Math.random() * 20;
            
            this.obstacles.push({
                x: x,
                y: y,
                radius: radius,
                type: 'rock'
            });
        }
        
        // Generate water patches
        const waterCount = Math.max(1, Math.floor(levelNum / 3));
        for (let i = 0; i < waterCount; i++) {
            const x = 200 + Math.random() * (this.width - 400);
            const y = 100 + Math.random() * (this.height - 200);
            const radius = 25 + Math.random() * 20;
            
            this.waterPatches.push({
                x: x,
                y: y,
                radius: radius
            });
        }
        
        // Reset spores for new level
        this.spores = 5 + levelNum * 2;
        this.updateUI();
    }
    
    setupEventListeners() {
        // Mouse controls for slingshot
        this.canvas.addEventListener('mousemove', (e) => {
            const rect = this.canvas.getBoundingClientRect();
            this.mousePos.x = e.clientX - rect.left;
            this.mousePos.y = e.clientY - rect.top;
            
            if (this.gameState === 'playing') {
                this.updateAim();
            }
        });
        
        this.canvas.addEventListener('click', (e) => {
            if (this.gameState === 'playing' && this.spores > 0) {
                this.launchSpore();
            }
        });
        
        // UI controls
        document.getElementById('reset-btn').addEventListener('click', () => {
            this.generateLevel(this.level);
        });
        
        document.getElementById('help-btn').addEventListener('click', () => {
            document.getElementById('help-modal').classList.remove('hidden');
        });
        
        document.getElementById('close-help').addEventListener('click', () => {
            document.getElementById('help-modal').classList.add('hidden');
        });
    }
    
    updateAim() {
        const dx = this.mousePos.x - this.slingshot.x;
        const dy = this.mousePos.y - this.slingshot.y;
        this.aimAngle = Math.atan2(dy, dx);
        this.aimPower = Math.min(Math.sqrt(dx * dx + dy * dy) / 3, 50);
    }
    
    launchSpore() {
        if (this.spores <= 0) return;
        
        const velocity = {
            x: Math.cos(this.aimAngle) * this.aimPower * 0.3,
            y: Math.sin(this.aimAngle) * this.aimPower * 0.3
        };
        
        this.projectiles.push({
            x: this.slingshot.x,
            y: this.slingshot.y,
            vx: velocity.x,
            vy: velocity.y,
            radius: 5,
            life: 300, // frames until spore dies
            trail: []
        });
        
        this.spores--;
        this.updateUI();
        
        // Add launch particles
        this.addParticles(this.slingshot.x, this.slingshot.y, 8, this.colors.spore);
    }
    
    updateProjectiles() {
        for (let i = this.projectiles.length - 1; i >= 0; i--) {
            const proj = this.projectiles[i];
            
            // Physics
            proj.x += proj.vx;
            proj.y += proj.vy;
            proj.vy += 0.15; // gravity
            proj.vx *= 0.999; // air resistance
            proj.life--;
            
            // Trail effect
            proj.trail.push({ x: proj.x, y: proj.y });
            if (proj.trail.length > 10) proj.trail.shift();
            
            // Check collisions
            this.checkProjectileCollisions(proj, i);
            
            // Remove if dead or out of bounds
            if (proj.life <= 0 || proj.x < 0 || proj.x > this.width || proj.y > this.height) {
                this.projectiles.splice(i, 1);
            }
        }
    }
    
    checkProjectileCollisions(proj, projIndex) {
        // Check soil patches
        for (let soil of this.soilPatches) {
            const dist = Math.sqrt((proj.x - soil.x) ** 2 + (proj.y - soil.y) ** 2);
            if (dist < soil.radius + proj.radius && !soil.hasMushroom) {
                this.growMushroom(soil);
                this.addParticles(proj.x, proj.y, 12, this.colors.mushroom);
                this.projectiles.splice(projIndex, 1);
                return;
            }
        }
        
        // Check obstacles (rocks)
        for (let obstacle of this.obstacles) {
            const dist = Math.sqrt((proj.x - obstacle.x) ** 2 + (proj.y - obstacle.y) ** 2);
            if (dist < obstacle.radius + proj.radius) {
                this.addParticles(proj.x, proj.y, 8, this.colors.rock);
                this.projectiles.splice(projIndex, 1);
                return;
            }
        }
        
        // Check water patches
        for (let water of this.waterPatches) {
            const dist = Math.sqrt((proj.x - water.x) ** 2 + (proj.y - water.y) ** 2);
            if (dist < water.radius + proj.radius) {
                this.addParticles(proj.x, proj.y, 6, this.colors.water);
                this.projectiles.splice(projIndex, 1);
                return;
            }
        }
    }
    
    growMushroom(soil) {
        soil.hasMushroom = true;
        this.mushrooms.push({
            x: soil.x,
            y: soil.y,
            radius: soil.radius * 0.8,
            growthPhase: 0,
            maxGrowth: 100
        });
        
        this.score += 100;
        this.updateUI();
        
        // Check for level completion
        if (this.mushrooms.length >= this.soilPatches.length) {
            setTimeout(() => this.nextLevel(), 1000);
        }
        
        // Chain reaction check (mushroom colonies)
        this.checkChainReaction(soil);
    }
    
    checkChainReaction(newMushroom) {
        // Check if this mushroom is close to other mushrooms for bonus points
        let nearbyMushrooms = 0;
        for (let soil of this.soilPatches) {
            if (soil !== newMushroom && soil.hasMushroom) {
                const dist = Math.sqrt((newMushroom.x - soil.x) ** 2 + (newMushroom.y - soil.y) ** 2);
                if (dist < 80) {
                    nearbyMushrooms++;
                }
            }
        }
        
        if (nearbyMushrooms > 0) {
            const bonus = nearbyMushrooms * 50;
            this.score += bonus;
            this.addParticles(newMushroom.x, newMushroom.y, 15, this.colors.mushroomSpot);
        }
    }
    
    nextLevel() {
        this.level++;
        this.generateLevel(this.level);
        this.gameState = 'playing';
    }
    
    addParticles(x, y, count, color) {
        for (let i = 0; i < count; i++) {
            this.particles.push({
                x: x,
                y: y,
                vx: (Math.random() - 0.5) * 8,
                vy: (Math.random() - 0.5) * 8,
                life: 30 + Math.random() * 20,
                maxLife: 50,
                color: color
            });
        }
    }
    
    updateParticles() {
        for (let i = this.particles.length - 1; i >= 0; i--) {
            const p = this.particles[i];
            p.x += p.vx;
            p.y += p.vy;
            p.vy += 0.1;
            p.vx *= 0.98;
            p.life--;
            
            if (p.life <= 0) {
                this.particles.splice(i, 1);
            }
        }
    }
    
    updateMushrooms() {
        for (let mushroom of this.mushrooms) {
            if (mushroom.growthPhase < mushroom.maxGrowth) {
                mushroom.growthPhase++;
            }
        }
    }
    
    render() {
        // Clear canvas
        this.ctx.fillStyle = this.colors.background;
        this.ctx.fillRect(0, 0, this.width, this.height);
        
        // Draw trajectory if aiming
        if (this.gameState === 'playing') {
            this.drawTrajectory();
        }
        
        // Draw game objects
        this.drawWaterPatches();
        this.drawSoilPatches();
        this.drawObstacles();
        this.drawMushrooms();
        this.drawProjectiles();
        this.drawSlingshot();
        this.drawParticles();
        this.drawUI();
    }
    
    drawTrajectory() {
        this.ctx.strokeStyle = this.colors.trajectory;
        this.ctx.lineWidth = 2;
        this.ctx.setLineDash([5, 5]);
        this.ctx.beginPath();
        
        // Simulate trajectory
        let x = this.slingshot.x;
        let y = this.slingshot.y;
        let vx = Math.cos(this.aimAngle) * this.aimPower * 0.3;
        let vy = Math.sin(this.aimAngle) * this.aimPower * 0.3;
        
        this.ctx.moveTo(x, y);
        for (let i = 0; i < 50; i++) {
            x += vx;
            y += vy;
            vy += 0.15;
            vx *= 0.999;
            this.ctx.lineTo(x, y);
        }
        
        this.ctx.stroke();
        this.ctx.setLineDash([]);
    }
    
    drawSoilPatches() {
        for (let soil of this.soilPatches) {
            this.ctx.fillStyle = this.colors.soil;
            this.ctx.beginPath();
            this.ctx.arc(soil.x, soil.y, soil.radius, 0, Math.PI * 2);
            this.ctx.fill();
            
            // Draw pixelated texture
            this.ctx.fillStyle = this.colors.soil + '80';
            for (let i = 0; i < 6; i++) {
                const px = soil.x + (Math.random() - 0.5) * soil.radius;
                const py = soil.y + (Math.random() - 0.5) * soil.radius;
                this.ctx.fillRect(px, py, 2, 2);
            }
        }
    }
    
    drawMushrooms() {
        for (let mushroom of this.mushrooms) {
            const growth = mushroom.growthPhase / mushroom.maxGrowth;
            const currentRadius = mushroom.radius * growth;
            
            // Mushroom stem
            this.ctx.fillStyle = '#F5F5DC';
            this.ctx.fillRect(
                mushroom.x - 3,
                mushroom.y,
                6,
                currentRadius * 0.8
            );
            
            // Mushroom cap
            this.ctx.fillStyle = this.colors.mushroom;
            this.ctx.beginPath();
            this.ctx.arc(mushroom.x, mushroom.y - currentRadius * 0.3, currentRadius, 0, Math.PI * 2);
            this.ctx.fill();
            
            // Mushroom spots (pixelated)
            this.ctx.fillStyle = this.colors.mushroomSpot;
            const spots = Math.floor(growth * 4);
            for (let i = 0; i < spots; i++) {
                const angle = (i / spots) * Math.PI * 2;
                const spotX = mushroom.x + Math.cos(angle) * currentRadius * 0.5;
                const spotY = mushroom.y - currentRadius * 0.3 + Math.sin(angle) * currentRadius * 0.3;
                this.ctx.fillRect(spotX - 2, spotY - 2, 4, 4);
            }
        }
    }
    
    drawObstacles() {
        for (let obstacle of this.obstacles) {
            this.ctx.fillStyle = this.colors.rock;
            this.ctx.beginPath();
            this.ctx.arc(obstacle.x, obstacle.y, obstacle.radius, 0, Math.PI * 2);
            this.ctx.fill();
            
            // Pixelated texture
            this.ctx.fillStyle = '#FFFFFF40';
            for (let i = 0; i < 8; i++) {
                const px = obstacle.x + (Math.random() - 0.5) * obstacle.radius;
                const py = obstacle.y + (Math.random() - 0.5) * obstacle.radius;
                this.ctx.fillRect(px, py, 1, 1);
            }
        }
    }
    
    drawWaterPatches() {
        for (let water of this.waterPatches) {
            this.ctx.fillStyle = this.colors.water;
            this.ctx.beginPath();
            this.ctx.arc(water.x, water.y, water.radius, 0, Math.PI * 2);
            this.ctx.fill();
            
            // Water animation (simple wave effect)
            this.ctx.fillStyle = '#87CEEB60';
            const time = Date.now() * 0.005;
            for (let i = 0; i < 3; i++) {
                const waveY = water.y + Math.sin(time + i) * 5;
                this.ctx.fillRect(water.x - water.radius, waveY, water.radius * 2, 2);
            }
        }
    }
    
    drawProjectiles() {
        for (let proj of this.projectiles) {
            // Draw trail
            this.ctx.strokeStyle = this.colors.spore + '60';
            this.ctx.lineWidth = 3;
            this.ctx.beginPath();
            for (let i = 0; i < proj.trail.length; i++) {
                const point = proj.trail[i];
                if (i === 0) {
                    this.ctx.moveTo(point.x, point.y);
                } else {
                    this.ctx.lineTo(point.x, point.y);
                }
            }
            this.ctx.stroke();
            
            // Draw spore
            this.ctx.fillStyle = this.colors.spore;
            this.ctx.beginPath();
            this.ctx.arc(proj.x, proj.y, proj.radius, 0, Math.PI * 2);
            this.ctx.fill();
            
            // Spore glow
            this.ctx.fillStyle = this.colors.spore + '40';
            this.ctx.beginPath();
            this.ctx.arc(proj.x, proj.y, proj.radius * 2, 0, Math.PI * 2);
            this.ctx.fill();
        }
    }
    
    drawSlingshot() {
        // Base
        this.ctx.fillStyle = this.colors.slingshot;
        this.ctx.fillRect(this.slingshot.x - 8, this.slingshot.y - 5, 16, 10);
        
        // Arms
        this.ctx.strokeStyle = this.colors.slingshot;
        this.ctx.lineWidth = 4;
        this.ctx.beginPath();
        this.ctx.moveTo(this.slingshot.x - 8, this.slingshot.y);
        this.ctx.lineTo(this.slingshot.x - 15, this.slingshot.y - 20);
        this.ctx.moveTo(this.slingshot.x + 8, this.slingshot.y);
        this.ctx.lineTo(this.slingshot.x + 15, this.slingshot.y - 20);
        this.ctx.stroke();
        
        // Elastic band when aiming
        if (this.gameState === 'playing' && this.aimPower > 0) {
            this.ctx.strokeStyle = this.colors.spore;
            this.ctx.lineWidth = 2;
            this.ctx.beginPath();
            this.ctx.moveTo(this.slingshot.x - 15, this.slingshot.y - 20);
            this.ctx.lineTo(this.mousePos.x, this.mousePos.y);
            this.ctx.lineTo(this.slingshot.x + 15, this.slingshot.y - 20);
            this.ctx.stroke();
        }
    }
    
    drawParticles() {
        for (let p of this.particles) {
            const alpha = p.life / p.maxLife;
            this.ctx.fillStyle = p.color + Math.floor(alpha * 255).toString(16).padStart(2, '0');
            this.ctx.fillRect(p.x - 1, p.y - 1, 2, 2);
        }
    }
    
    drawUI() {
        // Level completion message
        if (this.mushrooms.length >= this.soilPatches.length && this.soilPatches.length > 0) {
            this.ctx.fillStyle = 'rgba(0, 0, 0, 0.8)';
            this.ctx.fillRect(0, 0, this.width, this.height);
            
            this.ctx.fillStyle = this.colors.mushroom;
            this.ctx.font = 'bold 36px Orbitron, monospace';
            this.ctx.textAlign = 'center';
            this.ctx.fillText('LEVEL COMPLETE!', this.width / 2, this.height / 2 - 30);
            
            this.ctx.fillStyle = this.colors.spore;
            this.ctx.font = '24px Orbitron, monospace';
            this.ctx.fillText('🍄 All mushrooms grown! 🍄', this.width / 2, this.height / 2 + 20);
        }
        
        // Game over message
        if (this.spores <= 0 && this.projectiles.length === 0 && this.mushrooms.length < this.soilPatches.length) {
            this.ctx.fillStyle = 'rgba(0, 0, 0, 0.8)';
            this.ctx.fillRect(0, 0, this.width, this.height);
            
            this.ctx.fillStyle = '#FF0000';
            this.ctx.font = 'bold 36px Orbitron, monospace';
            this.ctx.textAlign = 'center';
            this.ctx.fillText('GAME OVER', this.width / 2, this.height / 2 - 30);
            
            this.ctx.fillStyle = this.colors.spore;
            this.ctx.font = '20px Orbitron, monospace';
            this.ctx.fillText('No more spores! Click Reset to try again.', this.width / 2, this.height / 2 + 20);
        }
    }
    
    updateUI() {
        document.getElementById('level').textContent = this.level;
        document.getElementById('score').textContent = this.score;
        document.getElementById('spores').textContent = this.spores;
    }
    
    gameLoop() {
        this.updateProjectiles();
        this.updateParticles();
        this.updateMushrooms();
        this.render();
        requestAnimationFrame(() => this.gameLoop());
    }
}

// Initialize game when page loads
window.addEventListener('load', () => {
    new MushroomGame();
});