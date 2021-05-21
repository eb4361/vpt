// #package glsl/shaders

// #include ../mixins/Photon.glsl
// #include ../mixins/rand.glsl
// #include ../mixins/unprojectRand.glsl
// #include ../mixins/intersectCube.glsl

// #section MCMGenerate/vertex

void main() {}

// #section MCMGenerate/fragment

void main() {}

// #section MCMIntegrate/vertex

#version 300 es

layout (location = 0) in vec2 aPosition;

out vec2 vPosition;

void main() {
    vPosition = aPosition;
    gl_Position = vec4(aPosition, 0.0, 1.0);
}

// #section MCMIntegrate/fragment

#version 300 es
precision mediump float;

#define M_INVPI 0.31830988618
#define M_2PI 6.28318530718
#define M_PI 3.1415926535897932384626433832795
#define EPS 1e-5

@Photon

uniform mediump sampler2D uPosition;
uniform mediump sampler2D uDirection;
uniform mediump sampler2D uTransmittance;
uniform mediump sampler2D uRadiance;

uniform mediump sampler3D uVolume;
uniform mediump sampler2D uTransferFunction;
uniform mediump sampler2D uEnvironment;

uniform mat4 uMvpInverseMatrix;
uniform vec2 uInverseResolution;
uniform float uRandSeed;
uniform float uBlur;

uniform float uAbsorptionCoefficient;
uniform float uScatteringCoefficient;
uniform float uScatteringBias;
uniform float uMajorant;
uniform uint uMaxBounces;
uniform uint uSteps;
uniform float uIsovalue;
uniform vec3 uBaseColor;
uniform vec3 uLight;
uniform float uMetallic;
uniform float uRoughness;

in vec2 vPosition;

layout (location = 0) out vec4 oPosition;
layout (location = 1) out vec4 oDirection;
layout (location = 2) out vec4 oTransmittance;
layout (location = 3) out vec4 oRadiance;

@rand
@unprojectRand
@intersectCube

void resetPhoton(inout vec2 randState, inout Photon photon) {
    vec3 from, to;
    unprojectRand(randState, vPosition, uMvpInverseMatrix, uInverseResolution, uBlur, from, to);
    photon.direction = normalize(to - from);
    photon.bounces = 0u;
    vec2 tbounds = max(intersectCube(from, photon.direction), 0.0);
    photon.position = from + tbounds.x * photon.direction;
    photon.transmittance = vec3(1);
}

vec4 sampleEnvironmentMap(vec3 d) {
    vec2 texCoord = vec2(atan(d.x, -d.z), asin(-d.y) * 2.0) * M_INVPI * 0.5 + 0.5;
    return texture(uEnvironment, texCoord);
}

vec4 sampleVolumeColor(vec3 position) {
    vec2 volumeSample = texture(uVolume, position).rg;
    vec4 transferSample = texture(uTransferFunction, volumeSample);
    return transferSample;
}

vec3 randomDirection(vec2 U) {
    float phi = U.x * M_2PI;
    float z = U.y * 2.0 - 1.0;
    float k = sqrt(1.0 - z * z);
    return vec3(k * cos(phi), k * sin(phi), z);
}

float sampleHenyeyGreensteinAngleCosine(float g, float U) {
    float g2 = g * g;
    float c = (1.0 - g2) / (1.0 - g + 2.0 * g * U);
    return (1.0 + g2 - c * c) / (2.0 * g);
}

vec3 sampleHenyeyGreenstein(float g, vec2 U, vec3 direction) {
    // generate random direction and adjust it so that the angle is HG-sampled
    vec3 u = randomDirection(U);
    if (abs(g) < EPS) {
        return u;
    }
    float hgcos = sampleHenyeyGreensteinAngleCosine(g, fract(sin(U.x * 12345.6789) + 0.816723));
    float lambda = hgcos - dot(direction, u);
    return normalize(u + lambda * direction);
}

vec3 gradient(vec3 pos, float h) {
    vec3 positive = vec3(
        texture(uVolume, pos + vec3( h, 0.0, 0.0)).r,
        texture(uVolume, pos + vec3(0.0,  h, 0.0)).r,
        texture(uVolume, pos + vec3(0.0, 0.0,  h)).r
    );
    vec3 negative = vec3(
        texture(uVolume, pos + vec3(-h, 0.0, 0.0)).r,
        texture(uVolume, pos + vec3(0.0, -h, 0.0)).r,
        texture(uVolume, pos + vec3(0.0, 0.0, -h)).r
    );
    return normalize(positive - negative);
}

float geometricOclusion(vec3 normal, vec3 viewVector, float alfa){
    return dot(normal, viewVector) + sqrt(pow(alfa, 2.0f) + (1.0f - pow(alfa, 2.0f) * pow(dot(normal,viewVector), 2.0f)));
}

vec3 importanceSampleGgxD(vec2 seed, float rough2, vec3 N)
{
    float phi = 2.0f * M_PI * seed.x;
    float cosTheta = sqrt((1.0f - seed.y) / (1.0f + (rough2*rough2 - 1.0f) * seed.y));
    float sinTheta = sqrt(1.0f - cosTheta*cosTheta);
    vec3 h;
    h.x = sinTheta * cos(phi);
    h.y = cosTheta;
    h.z = sinTheta * sin(phi);
    vec3 up = abs(N.y) < 0.999f ? vec3(0, 1, 0) : vec3(1, 0, 0);
    vec3 tangentX = normalize(cross(up, N));
    vec3 tangentZ = cross(tangentX, N);
    return h.x * tangentX + h.y * N + h.z * tangentZ;
}

void main() {
    Photon photon;
    vec2 mappedPosition = vPosition * 0.5 + 0.5;
    photon.position = texture(uPosition, mappedPosition).xyz;
    vec4 directionAndBounces = texture(uDirection, mappedPosition);
    photon.direction = directionAndBounces.xyz;
    photon.bounces = uint(directionAndBounces.w + 0.5);
    photon.transmittance = texture(uTransmittance, mappedPosition).rgb;
    vec4 radianceAndSamples = texture(uRadiance, mappedPosition);
    photon.radiance = radianceAndSamples.rgb;
    photon.samples = uint(radianceAndSamples.w + 0.5);

    vec2 r = rand(vPosition * uRandSeed);
    for (uint i = 0u; i < uSteps; i++) {
        r = rand(r);
        float t = -log(r.x) / uMajorant;
        vec3 firstPosition = photon.position;
        photon.position += t * photon.direction;
        vec3 secondPosition = photon.position;

        float value = texture(uVolume, photon.position).r;

        float secondTexturePosition = texture(uVolume, secondPosition).r;

        if(value >= uIsovalue){
            // Bisekcija
            vec3 middlePoint = firstPosition;
            if(texture(uVolume, firstPosition).r * secondTexturePosition >= 0.0f){
                // If this happens then there is a mistake
                continue;
            }
            else{
                if(value == uIsovalue) middlePoint = photon.position;
                else{
                    // 20 is  the limit so that we don't take too much time solving
                    for (int i = 0; i < 20; i++) {

                        middlePoint = (firstPosition + secondPosition) / 2.0f;
                        float middlePointTexture = texture(uVolume, middlePoint).r;

                        if (middlePointTexture * texture(uVolume, firstPosition).r >= 0.0f) firstPosition = middlePoint;
                        else secondPosition = middlePoint;
                    }
                }
            }
            
            // BRDF
            vec3 lightVector = normalize(uLight);
            vec3 viewVector = normalize(photon.direction);
            vec3 halfVector = normalize(lightVector + viewVector);
            vec3 normal = normalize(gradient(middlePoint.xyz, 0.005));

            // BRDF diffuse
            vec3 dielectric = vec3(0.04, 0.04, 0.04);
            vec3 colorBlack = vec3(0,0,0);
            vec3 cdiff = mix(uBaseColor.rgb * (1.0f-dielectric.r), colorBlack, uMetallic);
            vec3 diff = cdiff / M_PI;

            // BRDF Specular
            // Specular F
            vec3 F0 = mix(dielectric, uBaseColor.rgb, uMetallic);
            vec3 F = F0 + (1.0f-F0) * (pow(1.0f - abs(dot(viewVector, halfVector)), 5.0f));

            // Specular G
            float alfa = pow(uRoughness, 2.0f);
            float G1 = (2.0f * dot(normal, lightVector)) / geometricOclusion(normal, lightVector, alfa);
            float G2 = (2.0f * dot(normal, viewVector)) / geometricOclusion(normal, viewVector, alfa);
            float G = G1 * G2;

            // Specular D
            float belowD = M_PI * pow(pow(dot(normal, halfVector), 2.0f) * (alfa - 1.0f) + 1.0f, 2.0f);
            float D = pow(alfa, 2.0f) / belowD;

            // Specular final
            float belowSpec = 4.0f * dot(normal, lightVector) * dot(normal, viewVector);
            vec3 BRDFspec = F * G * D / belowSpec;
            vec3 BRDF = BRDFspec + diff;

            // sampling
            photon.position = middlePoint;

            photon.transmittance = photon.transmittance * BRDF;

            photon.direction = importanceSampleGgxD(r, alfa, normal);


        }
        else{
            vec4 volumeSample = sampleVolumeColor(photon.position);
            float muAbsorption = volumeSample.a * uAbsorptionCoefficient;
            float muScattering = volumeSample.a * uScatteringCoefficient;
            float muNull = uMajorant - muAbsorption - muScattering;
            float muMajorant = muAbsorption + muScattering + abs(muNull);
            float PNull = abs(muNull) / muMajorant;
            float PAbsorption = muAbsorption / muMajorant;
            float PScattering = muScattering / muMajorant;

            if (any(greaterThan(photon.position, vec3(1))) || any(lessThan(photon.position, vec3(0)))) {
                // out of bounds
                vec4 envSample = sampleEnvironmentMap(photon.direction);
                vec3 radiance = photon.transmittance * envSample.rgb;
                photon.samples++;
                photon.radiance += (radiance - photon.radiance) / float(photon.samples);
                resetPhoton(r, photon);
            } else if (photon.bounces >= uMaxBounces) {
                // max bounces achieved -> only estimate transmittance
                float weightAS = (muAbsorption + muScattering) / uMajorant;
                photon.transmittance *= 1.0 - weightAS;
            } else if (r.y < PAbsorption) {
                // absorption
                float weightA = muAbsorption / (uMajorant * PAbsorption);
                photon.transmittance *= 1.0 - weightA;
            } else if (r.y < PAbsorption + PScattering) {
                // scattering
                r = rand(r);
                float weightS = muScattering / (uMajorant * PScattering);
                photon.transmittance *= volumeSample.rgb * weightS;
                photon.direction = sampleHenyeyGreenstein(uScatteringBias, r, photon.direction);
                photon.bounces++;
            } else {
                // null collision
                float weightN = muNull / (uMajorant * PNull);
                photon.transmittance *= weightN;
            }
        }
    }

    oPosition = vec4(photon.position, 0);
    oDirection = vec4(photon.direction, float(photon.bounces));
    oTransmittance = vec4(photon.transmittance, 0);
    oRadiance = vec4(photon.radiance, float(photon.samples));
}

// #section MCMRender/vertex

#version 300 es

layout (location = 0) in vec2 aPosition;
out vec2 vPosition;

void main() {
    vPosition = (aPosition + 1.0) * 0.5;
    gl_Position = vec4(aPosition, 0.0, 1.0);
}

// #section MCMRender/fragment

#version 300 es
precision mediump float;

uniform mediump sampler2D uColor;

in vec2 vPosition;
out vec4 oColor;

void main() {
    oColor = vec4(texture(uColor, vPosition).rgb, 1);
}

// #section MCMReset/vertex

#version 300 es

layout (location = 0) in vec2 aPosition;

out vec2 vPosition;

void main() {
    vPosition = aPosition;
    gl_Position = vec4(aPosition, 0.0, 1.0);
}

// #section MCMReset/fragment

#version 300 es
precision mediump float;

@Photon

uniform mat4 uMvpInverseMatrix;
uniform vec2 uInverseResolution;
uniform float uRandSeed;
uniform float uBlur;

in vec2 vPosition;

layout (location = 0) out vec4 oPosition;
layout (location = 1) out vec4 oDirection;
layout (location = 2) out vec4 oTransmittance;
layout (location = 3) out vec4 oRadiance;

@rand
@unprojectRand
@intersectCube

void main() {
    Photon photon;
    vec3 from, to;
    vec2 randState = rand(vPosition * uRandSeed);
    unprojectRand(randState, vPosition, uMvpInverseMatrix, uInverseResolution, uBlur, from, to);
    photon.direction = normalize(to - from);
    vec2 tbounds = max(intersectCube(from, photon.direction), 0.0);
    photon.position = from + tbounds.x * photon.direction;
    photon.transmittance = vec3(1);
    photon.radiance = vec3(1);
    photon.bounces = 0u;
    photon.samples = 0u;
    oPosition = vec4(photon.position, 0);
    oDirection = vec4(photon.direction, float(photon.bounces));
    oTransmittance = vec4(photon.transmittance, 0);
    oRadiance = vec4(photon.radiance, float(photon.samples));
}
