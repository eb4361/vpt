// #package js/main

// #include ../AbstractDialog.js
// #include ../../TransferFunctionWidget.js

// #include ../../../uispecs/renderers/MCMRendererDialog.json

class MCMRendererDialog extends AbstractDialog {

    constructor(renderer, options) {
        super(UISPECS.MCMRendererDialog, options);
    
        this._renderer = renderer;
    
        this._handleChange = this._handleChange.bind(this);
        this._handleTFChange = this._handleTFChange.bind(this);
    
        this._binds.extinction.addEventListener('input', this._handleChange);
        this._binds.baseColor.addEventListener('change', this._handleChange);
        this._binds.metallic.addEventListener('input', this._handleChange);
        this._binds.roughness.addEventListener('input', this._handleChange);
        this._binds.isovalue.addEventListener('change', this._handleChange);
        this._binds.albedo.addEventListener('change', this._handleChange);
        this._binds.bias.addEventListener('change', this._handleChange);
        this._binds.ratio.addEventListener('change', this._handleChange);
        this._binds.bounces.addEventListener('input', this._handleChange);
        this._binds.steps.addEventListener('input', this._handleChange);
    
        this._tfwidget = new TransferFunctionWidget();
        this._binds.tfcontainer.add(this._tfwidget);
        this._tfwidget.addEventListener('change', this._handleTFChange);
    }
    
    destroy() {
        this._tfwidget.destroy();
        super.destroy();
    }
    
    _handleChange() {
        const extinction = this._binds.extinction.getValue();
        const baseColor  = CommonUtils.hex2rgb(this._binds.baseColor.getValue());
        const metallic   = this._binds.metallic.getValue();
        const roughness  = this._binds.roughness.getValue();
        const isovalue   = this._binds.isovalue.getValue();
        const albedo     = this._binds.albedo.getValue();
        const bias       = this._binds.bias.getValue();
        const ratio      = this._binds.ratio.getValue();
        const bounces    = this._binds.bounces.getValue();
        const steps      = this._binds.steps.getValue();
    
        this._renderer.absorptionCoefficient = extinction * (1 - albedo);
        this._renderer.scatteringCoefficient = extinction * albedo;
        this._renderer.scatteringBias = bias;
        this._renderer.majorant = extinction * ratio;
        this._renderer.maxBounces = bounces;
        this._renderer.steps = steps;
        this._renderer.isovalue = isovalue;
        this._renderer._baseColor[0] = baseColor.r;
        this._renderer._baseColor[1] = baseColor.g;
        this._renderer._baseColor[2] = baseColor.b;
        this._renderer.metallic = metallic;
        this._renderer.roughness = roughness;
    
        console.log("baseColorR: " + baseColor.r);
        console.log("baseColorR: " + baseColor.g);
        console.log("baseColorR: " + baseColor.b);
        console.log("metallic: " + metallic);
        console.log("roughness: " + roughness);
    
        this._renderer.reset();
    }
    
    _handleTFChange() {
        this._renderer.setTransferFunction(this._tfwidget.getTransferFunction());
        this._renderer.reset();
    }
    
    }
    