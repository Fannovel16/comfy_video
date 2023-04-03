import { app } from "/scripts/app.js";
import { ComfyWidgets } from "/scripts/widgets.js";

function getWidgetType(config) {
	// Special handling for COMBO so we restrict links based on the entries
	let type = config[0];
	let linkType = type;
	if (type instanceof Array) {
		type = "COMBO";
		linkType = linkType.join(",");
	}
	return { type, linkType };
}

class AutoIncrementor {
    constructor() {
        this.isVirtualNode = true;
        this.serialize_widgets = true;
        const indexWidget = ComfyWidgets.INT(this, "index", [undefined, { default: 1, min: 1, max: Number.MAX_SAFE_INTEGER }])
        indexWidget.afterQueued = function () { this.value++ }
        this.addOutput("connect to INT widget input", "INT");
    }
    applyToGraph () {
        if (!this.outputs[0].links?.length) return;

        // For each output link copy our value over the original widget value
        for (const l of this.outputs[0].links) {
            const linkInfo = app.graph.links[l];
            const node = this.graph.getNodeById(linkInfo.target_id);
            const input = node.inputs[linkInfo.target_slot];
            const widgetName = input.widget.name;
            if (widgetName) {
                const widget = node.widgets.find((w) => w.name === widgetName);
                if (widget) {
                    widget.value = this.widgets[0].value;
                    if (widget.callback) {
                        widget.callback(widget.value, app.canvas, node, app.canvas.graph_mouse, {});
                    }
                }
            }
        }
    }
    onConnectionsChange(_, index, connected) {
        if (connected && this.outputs[0].links?.length) {
            // First connection can fire before the graph is ready on initial load so random things can be missing
            const linkId = this.outputs[0].links[0];
            const link = this.graph.links[linkId];
            if (!link) return;

            const theirNode = this.graph.getNodeById(link.target_id);
            if (!theirNode || !theirNode.inputs) return;

            const input = theirNode.inputs[link.target_slot];
            if (!input) return;

            const widget = input.widget;
            console.log(widget.config)
            const { type, linkType } = getWidgetType(widget.config);

            // Update our output to restrict to the widget type
            this.outputs[0].type = linkType;
            this.outputs[0].name = widget.name;
            this.outputs[0].widget = widget;
            return
        }
        if (!this.outputs[0].links?.length) {
            // We cant remove + re-add the output here as if you drag a link over the same link
            // it removes, then re-adds, causing it to break
            this.outputs[0].type = "INT";
            this.outputs[0].name = "connect to INT widget input";
        }
    }
    
}

app.registerExtension({
    name: "Fannovel16.comfy_video",
    registerCustomNodes(app) {
        AutoIncrementor.title = "AutoIncrementor"
        LiteGraph.registerNodeType("AutoIncrementor", AutoIncrementor);
        AutoIncrementor.category = "utils"
    }
});
