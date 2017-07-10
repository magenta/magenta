/*!!
 *  p5.svg v0.6.0.0
 *  SVG Runtime for p5.js.
 *
 *  Copyright (C) 2015-2016 Zeno Zeng
 *  Licensed under the LGPL license.
 */
(function (root, factory) {
    if (typeof define === 'function' && define.amd) {
        define('p5.svg', ['p5'], function (p5) {
            factory(p5);
        });
    }
    else if (typeof exports === 'object') {
        module.exports = factory;
    }
    else {
        factory(root['p5']);
    }
})(this, function (p5) {

(function e(t,n,r){function s(o,u){if(!n[o]){if(!t[o]){var a=typeof require=="function"&&require;if(!u&&a)return a(o,!0);if(i)return i(o,!0);var f=new Error("Cannot find module '"+o+"'");throw f.code="MODULE_NOT_FOUND",f}var l=n[o]={exports:{}};t[o][0].call(l.exports,function(e){var n=t[o][1][e];return s(n?n:e)},l,l.exports,e,t,n,r)}return n[o].exports}var i=typeof require=="function"&&require;for(var o=0;o<r.length;o++)s(r[o]);return s})({1:[function(require,module,exports){
/*!!
 *  Canvas 2 Svg v1.0.9
 *  A low level canvas to SVG converter. Uses a mock canvas context to build an SVG document.
 *
 *  Licensed under the MIT license:
 *  http://www.opensource.org/licenses/mit-license.php
 *
 *  Author:
 *  Kerry Liu
 *
 *  Copyright (c) 2014 Gliffy Inc.
 */

;(function() {
    "use strict";

    var STYLES, ctx, CanvasGradient, CanvasPattern, namedEntities;

    //helper function to format a string
    function format(str, args) {
        var keys = Object.keys(args), i;
        for (i=0; i<keys.length; i++) {
            str = str.replace(new RegExp("\\{" + keys[i] + "\\}", "gi"), args[keys[i]]);
        }
        return str;
    }

    //helper function that generates a random string
    function randomString(holder) {
        var chars, randomstring, i;
        if (!holder) {
            throw new Error("cannot create a random attribute name for an undefined object");
        }
        chars = "ABCDEFGHIJKLMNOPQRSTUVWXTZabcdefghiklmnopqrstuvwxyz";
        randomstring = "";
        do {
            randomstring = "";
            for (i = 0; i < 12; i++) {
                randomstring += chars[Math.floor(Math.random() * chars.length)];
            }
        } while (holder[randomstring]);
        return randomstring;
    }

    //helper function to map named to numbered entities
    function createNamedToNumberedLookup(items, radix) {
        var i, entity, lookup = {}, base10, base16;
        items = items.split(',');
        radix = radix || 10;
        // Map from named to numbered entities.
        for (i = 0; i < items.length; i += 2) {
            entity = '&' + items[i + 1] + ';';
            base10 = parseInt(items[i], radix);
            lookup[entity] = '&#'+base10+';';
        }
        //FF and IE need to create a regex from hex values ie &nbsp; == \xa0
        lookup["\\xa0"] = '&#160;';
        return lookup;
    }

    //helper function to map canvas-textAlign to svg-textAnchor
    function getTextAnchor(textAlign) {
        //TODO: support rtl languages
        var mapping = {"left":"start", "right":"end", "center":"middle", "start":"start", "end":"end"};
        return mapping[textAlign] || mapping.start;
    }

    //helper function to map canvas-textBaseline to svg-dominantBaseline
    function getDominantBaseline(textBaseline) {
        //INFO: not supported in all browsers
        var mapping = {"alphabetic": "alphabetic", "hanging": "hanging", "top":"text-before-edge", "bottom":"text-after-edge", "middle":"central"};
        return mapping[textBaseline] || mapping.alphabetic;
    }

    // Unpack entities lookup where the numbers are in radix 32 to reduce the size
    // entity mapping courtesy of tinymce
    namedEntities = createNamedToNumberedLookup(
        '50,nbsp,51,iexcl,52,cent,53,pound,54,curren,55,yen,56,brvbar,57,sect,58,uml,59,copy,' +
            '5a,ordf,5b,laquo,5c,not,5d,shy,5e,reg,5f,macr,5g,deg,5h,plusmn,5i,sup2,5j,sup3,5k,acute,' +
            '5l,micro,5m,para,5n,middot,5o,cedil,5p,sup1,5q,ordm,5r,raquo,5s,frac14,5t,frac12,5u,frac34,' +
            '5v,iquest,60,Agrave,61,Aacute,62,Acirc,63,Atilde,64,Auml,65,Aring,66,AElig,67,Ccedil,' +
            '68,Egrave,69,Eacute,6a,Ecirc,6b,Euml,6c,Igrave,6d,Iacute,6e,Icirc,6f,Iuml,6g,ETH,6h,Ntilde,' +
            '6i,Ograve,6j,Oacute,6k,Ocirc,6l,Otilde,6m,Ouml,6n,times,6o,Oslash,6p,Ugrave,6q,Uacute,' +
            '6r,Ucirc,6s,Uuml,6t,Yacute,6u,THORN,6v,szlig,70,agrave,71,aacute,72,acirc,73,atilde,74,auml,' +
            '75,aring,76,aelig,77,ccedil,78,egrave,79,eacute,7a,ecirc,7b,euml,7c,igrave,7d,iacute,7e,icirc,' +
            '7f,iuml,7g,eth,7h,ntilde,7i,ograve,7j,oacute,7k,ocirc,7l,otilde,7m,ouml,7n,divide,7o,oslash,' +
            '7p,ugrave,7q,uacute,7r,ucirc,7s,uuml,7t,yacute,7u,thorn,7v,yuml,ci,fnof,sh,Alpha,si,Beta,' +
            'sj,Gamma,sk,Delta,sl,Epsilon,sm,Zeta,sn,Eta,so,Theta,sp,Iota,sq,Kappa,sr,Lambda,ss,Mu,' +
            'st,Nu,su,Xi,sv,Omicron,t0,Pi,t1,Rho,t3,Sigma,t4,Tau,t5,Upsilon,t6,Phi,t7,Chi,t8,Psi,' +
            't9,Omega,th,alpha,ti,beta,tj,gamma,tk,delta,tl,epsilon,tm,zeta,tn,eta,to,theta,tp,iota,' +
            'tq,kappa,tr,lambda,ts,mu,tt,nu,tu,xi,tv,omicron,u0,pi,u1,rho,u2,sigmaf,u3,sigma,u4,tau,' +
            'u5,upsilon,u6,phi,u7,chi,u8,psi,u9,omega,uh,thetasym,ui,upsih,um,piv,812,bull,816,hellip,' +
            '81i,prime,81j,Prime,81u,oline,824,frasl,88o,weierp,88h,image,88s,real,892,trade,89l,alefsym,' +
            '8cg,larr,8ch,uarr,8ci,rarr,8cj,darr,8ck,harr,8dl,crarr,8eg,lArr,8eh,uArr,8ei,rArr,8ej,dArr,' +
            '8ek,hArr,8g0,forall,8g2,part,8g3,exist,8g5,empty,8g7,nabla,8g8,isin,8g9,notin,8gb,ni,8gf,prod,' +
            '8gh,sum,8gi,minus,8gn,lowast,8gq,radic,8gt,prop,8gu,infin,8h0,ang,8h7,and,8h8,or,8h9,cap,8ha,cup,' +
            '8hb,int,8hk,there4,8hs,sim,8i5,cong,8i8,asymp,8j0,ne,8j1,equiv,8j4,le,8j5,ge,8k2,sub,8k3,sup,8k4,' +
            'nsub,8k6,sube,8k7,supe,8kl,oplus,8kn,otimes,8l5,perp,8m5,sdot,8o8,lceil,8o9,rceil,8oa,lfloor,8ob,' +
            'rfloor,8p9,lang,8pa,rang,9ea,loz,9j0,spades,9j3,clubs,9j5,hearts,9j6,diams,ai,OElig,aj,oelig,b0,' +
            'Scaron,b1,scaron,bo,Yuml,m6,circ,ms,tilde,802,ensp,803,emsp,809,thinsp,80c,zwnj,80d,zwj,80e,lrm,' +
            '80f,rlm,80j,ndash,80k,mdash,80o,lsquo,80p,rsquo,80q,sbquo,80s,ldquo,80t,rdquo,80u,bdquo,810,dagger,' +
            '811,Dagger,81g,permil,81p,lsaquo,81q,rsaquo,85c,euro', 32);


    //Some basic mappings for attributes and default values.
    STYLES = {
        "strokeStyle":{
            svgAttr : "stroke", //corresponding svg attribute
            canvas : "#000000", //canvas default
            svg : "none",       //svg default
            apply : "stroke"    //apply on stroke() or fill()
        },
        "fillStyle":{
            svgAttr : "fill",
            canvas : "#000000",
            svg : null, //svg default is black, but we need to special case this to handle canvas stroke without fill
            apply : "fill"
        },
        "lineCap":{
            svgAttr : "stroke-linecap",
            canvas : "butt",
            svg : "butt",
            apply : "stroke"
        },
        "lineJoin":{
            svgAttr : "stroke-linejoin",
            canvas : "miter",
            svg : "miter",
            apply : "stroke"
        },
        "miterLimit":{
            svgAttr : "stroke-miterlimit",
            canvas : 10,
            svg : 4,
            apply : "stroke"
        },
        "lineWidth":{
            svgAttr : "stroke-width",
            canvas : 1,
            svg : 1,
            apply : "stroke"
        },
        "globalAlpha": {
            svgAttr : "opacity",
            canvas : 1,
            svg : 1,
            apply : "fill stroke"
        },
        "font":{
            //font converts to multiple svg attributes, there is custom logic for this
            canvas : "10px sans-serif"
        },
        "shadowColor":{
            canvas : "#000000"
        },
        "shadowOffsetX":{
            canvas : 0
        },
        "shadowOffsetY":{
            canvas : 0
        },
        "shadowBlur":{
            canvas : 0
        },
        "textAlign":{
            canvas : "start"
        },
        "textBaseline":{
            canvas : "alphabetic"
        }
    };

    /**
     *
     * @param gradientNode - reference to the gradient
     * @constructor
     */
    CanvasGradient = function(gradientNode) {
        this.__root = gradientNode;
    };

    /**
     * Adds a color stop to the gradient root
     */
    CanvasGradient.prototype.addColorStop = function(offset, color) {
        var stop = this.__createElement("stop"), regex, matches;
        stop.setAttribute("offset", offset);
        if(color.indexOf("rgba") !== -1) {
            //separate alpha value, since webkit can't handle it
            regex = /rgba\(\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d?\.?\d*)\s*\)/gi;
            matches = regex.exec(color);
            stop.setAttribute("stop-color", format("rgb({r},{g},{b})", {r:matches[1], g:matches[2], b:matches[3]}));
            stop.setAttribute("stop-opacity", matches[4]);
        } else {
            stop.setAttribute("stop-color", color);
        }
        this.__root.appendChild(stop);
    };

    CanvasPattern = function(pattern, ctx) {
        this.__root = pattern;
        this.__ctx = ctx;
    };

    /**
     * The mock canvas context
     * @param o - options include:
     * width - width of your canvas (defaults to 500)
     * height - height of your canvas (defaults to 500)
     * enableMirroring - enables canvas mirroring (get image data) (defaults to false)
     */
    ctx = function(o) {

        var defaultOptions = { width:500, height:500, enableMirroring : false }, options;

        //keep support for this way of calling C2S: new C2S(width,height)
        if(arguments.length > 1) {
            options = defaultOptions;
            options.width = arguments[0];
            options.height = arguments[1];
        } else if( !o ) {
            options = defaultOptions;
        } else {
            options = o;
        }

        if(!(this instanceof ctx)) {
            //did someone call this without new?
            return new ctx(options);
        }

        //setup options
        this.width = options.width || defaultOptions.width;
        this.height = options.height || defaultOptions.height;
        this.enableMirroring = options.enableMirroring !== undefined ? options.enableMirroring : defaultOptions.enableMirroring;

        this.canvas = this;   ///point back to this instance!
        this.__canvas = document.createElement("canvas");
        this.__ctx = this.__canvas.getContext("2d");

        this.__setDefaultStyles();
        this.__stack = [this.__getStyleState()];
        this.__groupStack = [];

        //the root svg element
        this.__root = document.createElementNS("http://www.w3.org/2000/svg", "svg");
        this.__root.setAttribute("version", 1.1);
        this.__root.setAttribute("xmlns", "http://www.w3.org/2000/svg");
        this.__root.setAttributeNS("http://www.w3.org/2000/xmlns/", "xmlns:xlink", "http://www.w3.org/1999/xlink");
        this.__root.setAttribute("width", this.width);
        this.__root.setAttribute("height", this.height);

        //make sure we don't generate the same ids in defs
        this.__ids = {};

        //defs tag
        this.__defs = document.createElementNS("http://www.w3.org/2000/svg", "defs");
        this.__root.appendChild(this.__defs);

        //also add a group child. the svg element can't use the transform attribute
        this.__currentElement = document.createElementNS("http://www.w3.org/2000/svg", "g");
        this.__root.appendChild(this.__currentElement);
    };

    /**
     * Creates the specified svg element
     * @private
     */
    ctx.prototype.__createElement = function(elementName, properties, resetFill) {
        if (typeof properties === "undefined") {
            properties = {};
        }

        var element = document.createElementNS("http://www.w3.org/2000/svg", elementName),
            keys = Object.keys(properties), i, key;
        if(resetFill) {
            //if fill or stroke is not specified, the svg element should not display. By default SVG's fill is black.
            element.setAttribute("fill", "none");
            element.setAttribute("stroke", "none");
        }
        for(i=0; i<keys.length; i++) {
            key = keys[i];
            element.setAttribute(key, properties[key]);
        }
        return element;
    };

    /**
     * Applies default canvas styles to the context
     * @private
     */
    ctx.prototype.__setDefaultStyles = function() {
        //default 2d canvas context properties see:http://www.w3.org/TR/2dcontext/
        var keys = Object.keys(STYLES), i, key;
        for(i=0; i<keys.length; i++) {
            key = keys[i];
            this[key] = STYLES[key].canvas;
        }
    };

    /**
     * Applies styles on restore
     * @param styleState
     * @private
     */
    ctx.prototype.__applyStyleState = function(styleState) {
        var keys = Object.keys(styleState), i, key;
        for(i=0; i<keys.length; i++) {
            key = keys[i];
            this[key] = styleState[key];
        }
    };

    /**
     * Gets the current style state
     * @return {Object}
     * @private
     */
    ctx.prototype.__getStyleState = function() {
        var i, styleState = {}, keys = Object.keys(STYLES), key;
        for(i=0; i<keys.length; i++) {
            key = keys[i];
            styleState[key] = this[key];
        }
        return styleState;
    };

    /**
     * Apples the current styles to the current SVG element. On "ctx.fill" or "ctx.stroke"
     * @param type
     * @private
     */
    ctx.prototype.__applyStyleToCurrentElement = function(type) {
        var keys = Object.keys(STYLES), i, style, value, id, regex, matches;
        for(i=0; i<keys.length; i++) {
            style = STYLES[keys[i]];
            value = this[keys[i]];
            if(style.apply) {
                //is this a gradient or pattern?
                if(style.apply.indexOf("fill")!==-1 && value instanceof CanvasPattern) {
                    //pattern
                    if(value.__ctx) {
                        //copy over defs
                        while(value.__ctx.__defs.childNodes.length) {
                            id = value.__ctx.__defs.childNodes[0].getAttribute("id");
                            this.__ids[id] = id;
                            this.__defs.appendChild(value.__ctx.__defs.childNodes[0]);
                        }
                    }
                    this.__currentElement.setAttribute("fill", format("url(#{id})", {id:value.__root.getAttribute("id")}));
                }
                else if(style.apply.indexOf("fill")!==-1 && value instanceof CanvasGradient) {
                    //gradient
                    this.__currentElement.setAttribute("fill", format("url(#{id})", {id:value.__root.getAttribute("id")}));
                } else if(style.apply.indexOf(type)!==-1 && style.svg !== value) {
                    if((style.svgAttr === "stroke" || style.svgAttr === "fill") && value.indexOf("rgba") !== -1) {
                        //separate alpha value, since illustrator can't handle it
                        regex = /rgba\(\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d?\.?\d*)\s*\)/gi;
                        matches = regex.exec(value);
                        this.__currentElement.setAttribute(style.svgAttr, format("rgb({r},{g},{b})", {r:matches[1], g:matches[2], b:matches[3]}));
                        this.__currentElement.setAttribute(style.svgAttr+"-opacity", matches[4]);
                    } else {
                        //otherwise only update attribute if right type, and not svg default
                        this.__currentElement.setAttribute(style.svgAttr, value);
                    }
                }
            }
        }

    };

    /**
     * Will return the closest group or svg node. May return the current element.
     * @private
     */
    ctx.prototype.__closestGroupOrSvg = function(node) {
        node = node || this.__currentElement;
        if(node.nodeName === "g" || node.nodeName === "svg") {
            return node;
        } else {
            return this.__closestGroupOrSvg(node.parentNode);
        }
    };

    /**
     * Returns the serialized value of the svg so far
     * @param fixNamedEntities - Standalone SVG doesn't support named entities, which document.createTextNode encodes.
     *                           If true, we attempt to find all named entities and encode it as a numeric entity.
     * @return serialized svg
     */
    ctx.prototype.getSerializedSvg = function(fixNamedEntities) {
        var serialized = new XMLSerializer().serializeToString(this.__root),
            keys, i, key, value, regexp, xmlns;

        //IE search for a duplicate xmnls because they didn't implement setAttributeNS correctly
        xmlns = /xmlns="http:\/\/www\.w3\.org\/2000\/svg".+xmlns="http:\/\/www\.w3\.org\/2000\/svg/gi;
        if(xmlns.test(serialized)) {
            serialized = serialized.replace('xmlns="http://www.w3.org/2000/svg','xmlns:xlink="http://www.w3.org/1999/xlink');
        }

        if(fixNamedEntities) {
            keys = Object.keys(namedEntities);
            //loop over each named entity and replace with the proper equivalent.
            for(i=0; i<keys.length; i++) {
                key = keys[i];
                value = namedEntities[key];
                regexp = new RegExp(key, "gi");
                if(regexp.test(serialized)) {
                    serialized = serialized.replace(regexp, value);
                }
            }
        }

        return serialized;
    };


    /**
     * Returns the root svg
     * @return
     */
    ctx.prototype.getSvg = function() {
        return this.__root;
    };
    /**
     * Will generate a group tag.
     */
    ctx.prototype.save = function() {
        var group = this.__createElement("g"), parent = this.__closestGroupOrSvg();
        this.__groupStack.push(parent);
        parent.appendChild(group);
        this.__currentElement = group;
        this.__stack.push(this.__getStyleState());
    };
    /**
     * Sets current element to parent, or just root if already root
     */
    ctx.prototype.restore = function(){
        this.__currentElement = this.__groupStack.pop();
        var state = this.__stack.pop();
        this.__applyStyleState(state);

    };

    /**
     * Helper method to add transform
     * @private
     */
    ctx.prototype.__addTransform = function(t) {

        //if the current element has siblings, add another group
        var parent = this.__closestGroupOrSvg();
        if(parent.childNodes.length > 0) {
            var group = this.__createElement("g");
            parent.appendChild(group);
            this.__currentElement = group;
        }

        var transform = this.__currentElement.getAttribute("transform");
        if(transform) {
            transform += " ";
        } else {
            transform = "";
        }
        transform += t;
        this.__currentElement.setAttribute("transform", transform);
    };

    /**
     *  scales the current element
     */
    ctx.prototype.scale = function(x, y) {
        if(y === undefined) {
            y = x;
        }
        this.__addTransform(format("scale({x},{y})", {x:x, y:y}));
    };

    /**
     * rotates the current element
     */
    ctx.prototype.rotate = function(angle){
        var degrees = (angle * 180 / Math.PI);
        this.__addTransform(format("rotate({angle},{cx},{cy})", {angle:degrees, cx:0, cy:0}));
    };

    /**
     * translates the current element
     */
    ctx.prototype.translate = function(x, y){
        this.__addTransform(format("translate({x},{y})", {x:x,y:y}));
    };

    /**
     * applies a transform to the current element
     */
    ctx.prototype.transform = function(a, b, c, d, e, f){
        this.__addTransform(format("matrix({a},{b},{c},{d},{e},{f})", {a:a, b:b, c:c, d:d, e:e, f:f}));
    };

    /**
     * Create a new Path Element
     */
    ctx.prototype.beginPath = function(){
        var path, parent;

        // Note that there is only one current default path, it is not part of the drawing state.
        // See also: https://html.spec.whatwg.org/multipage/scripting.html#current-default-path
        this.__currentDefaultPath = "";
        this.__currentPosition = {};

        path = this.__createElement("path", {}, true);
        parent = this.__closestGroupOrSvg();
        parent.appendChild(path);
        this.__currentElement = path;
    };

    /**
     * Helper function to apply currentDefaultPath to current path element
     * @private
     */
    ctx.prototype.__applyCurrentDefaultPath = function() {
        if(this.__currentElement.nodeName === "path") {
            var d = this.__currentDefaultPath;
            this.__currentElement.setAttribute("d", d);
        } else {
            throw new Error("Attempted to apply path command to node " + this.__currentElement.nodeName);
        }
    };

    /**
     * Helper function to add path command
     * @private
     */
    ctx.prototype.__addPathCommand = function(command){
        this.__currentDefaultPath += " ";
        this.__currentDefaultPath += command;
    };

    /**
     * Adds the move command to the current path element,
     * if the currentPathElement is not empty create a new path element
     */
    ctx.prototype.moveTo = function(x,y){
        if(this.__currentElement.nodeName !== "path") {
            this.beginPath();
        }

        // creates a new subpath with the given point
        this.__currentPosition = {x: x, y: y};
        this.__addPathCommand(format("M {x} {y}", {x:x, y:y}));
    };

    /**
     * Closes the current path
     */
    ctx.prototype.closePath = function(){
        this.__addPathCommand("Z");
    };

    /**
     * Adds a line to command
     */
    ctx.prototype.lineTo = function(x, y){
        this.__currentPosition = {x: x, y: y};
        if (this.__currentDefaultPath.indexOf('M') > -1) {
            this.__addPathCommand(format("L {x} {y}", {x:x, y:y}));
        } else {
            this.__addPathCommand(format("M {x} {y}", {x:x, y:y}));
        }
    };

    /**
     * Add a bezier command
     */
    ctx.prototype.bezierCurveTo = function(cp1x, cp1y, cp2x, cp2y, x, y) {
        this.__currentPosition = {x: x, y: y};
        this.__addPathCommand(format("C {cp1x} {cp1y} {cp2x} {cp2y} {x} {y}",
            {cp1x:cp1x, cp1y:cp1y, cp2x:cp2x, cp2y:cp2y, x:x, y:y}));
    };

    /**
     * Adds a quadratic curve to command
     */
    ctx.prototype.quadraticCurveTo = function(cpx, cpy, x, y){
        this.__currentPosition = {x: x, y: y};
        this.__addPathCommand(format("Q {cpx} {cpy} {x} {y}", {cpx:cpx, cpy:cpy, x:x, y:y}));
    };


    /**
     * Return a new normalized vector of given vector
     */
    var normalize = function(vector) {
        var len = Math.sqrt(vector[0] * vector[0] + vector[1] * vector[1]);
        return [vector[0] / len, vector[1] / len];
    };

    /**
     * Adds the arcTo to the current path
     *
     * @see http://www.w3.org/TR/2015/WD-2dcontext-20150514/#dom-context-2d-arcto
     */
    ctx.prototype.arcTo = function(x1, y1, x2, y2, radius) {
        // Let the point (x0, y0) be the last point in the subpath.
        var x0 = this.__currentPosition && this.__currentPosition.x;
        var y0 = this.__currentPosition && this.__currentPosition.y;

        // First ensure there is a subpath for (x1, y1).
        if (typeof x0 == "undefined" || typeof y0 == "undefined") {
            return;
        }

        // Negative values for radius must cause the implementation to throw an IndexSizeError exception.
        if (radius < 0) {
            throw new Error("IndexSizeError: The radius provided (" + radius + ") is negative.");
        }

        // If the point (x0, y0) is equal to the point (x1, y1),
        // or if the point (x1, y1) is equal to the point (x2, y2),
        // or if the radius radius is zero,
        // then the method must add the point (x1, y1) to the subpath,
        // and connect that point to the previous point (x0, y0) by a straight line.
        if (((x0 === x1) && (y0 === y1))
            || ((x1 === x2) && (y1 === y2))
            || (radius === 0)) {
            this.lineTo(x1, y1);
            return;
        }

        // Otherwise, if the points (x0, y0), (x1, y1), and (x2, y2) all lie on a single straight line,
        // then the method must add the point (x1, y1) to the subpath,
        // and connect that point to the previous point (x0, y0) by a straight line.
        var unit_vec_p1_p0 = normalize([x0 - x1, y0 - y1]);
        var unit_vec_p1_p2 = normalize([x2 - x1, y2 - y1]);
        if (unit_vec_p1_p0[0] * unit_vec_p1_p2[1] === unit_vec_p1_p0[1] * unit_vec_p1_p2[0]) {
            this.lineTo(x1, y1);
            return;
        }

        // Otherwise, let The Arc be the shortest arc given by circumference of the circle that has radius radius,
        // and that has one point tangent to the half-infinite line that crosses the point (x0, y0) and ends at the point (x1, y1),
        // and that has a different point tangent to the half-infinite line that ends at the point (x1, y1), and crosses the point (x2, y2).
        // The points at which this circle touches these two lines are called the start and end tangent points respectively.

        // note that both vectors are unit vectors, so the length is 1
        var cos = (unit_vec_p1_p0[0] * unit_vec_p1_p2[0] + unit_vec_p1_p0[1] * unit_vec_p1_p2[1]);
        var theta = Math.acos(Math.abs(cos));

        // Calculate origin
        var unit_vec_p1_origin = normalize([
            unit_vec_p1_p0[0] + unit_vec_p1_p2[0],
            unit_vec_p1_p0[1] + unit_vec_p1_p2[1]
        ]);
        var len_p1_origin = radius / Math.sin(theta / 2);
        var x = x1 + len_p1_origin * unit_vec_p1_origin[0];
        var y = y1 + len_p1_origin * unit_vec_p1_origin[1];

        // Calculate start angle and end angle
        // rotate 90deg clockwise (note that y axis points to its down)
        var unit_vec_origin_start_tangent = [
            -unit_vec_p1_p0[1],
            unit_vec_p1_p0[0]
        ];
        // rotate 90deg counter clockwise (note that y axis points to its down)
        var unit_vec_origin_end_tangent = [
            unit_vec_p1_p2[1],
            -unit_vec_p1_p2[0]
        ];
        var getAngle = function(vector) {
            // get angle (clockwise) between vector and (1, 0)
            var x = vector[0];
            var y = vector[1];
            if (y >= 0) { // note that y axis points to its down
                return Math.acos(x);
            } else {
                return -Math.acos(x);
            }
        };
        var startAngle = getAngle(unit_vec_origin_start_tangent);
        var endAngle = getAngle(unit_vec_origin_end_tangent);

        // Connect the point (x0, y0) to the start tangent point by a straight line
        this.lineTo(x + unit_vec_origin_start_tangent[0] * radius,
                    y + unit_vec_origin_start_tangent[1] * radius);

        // Connect the start tangent point to the end tangent point by arc
        // and adding the end tangent point to the subpath.
        this.arc(x, y, radius, startAngle, endAngle);
    };

    /**
     * Sets the stroke property on the current element
     */
    ctx.prototype.stroke = function(){
        if(this.__currentElement.nodeName === "path") {
            this.__currentElement.setAttribute("paint-order", "fill stroke markers");
        }
        this.__applyCurrentDefaultPath();
        this.__applyStyleToCurrentElement("stroke");
    };

    /**
     * Sets fill properties on the current element
     */
    ctx.prototype.fill = function(){
        if(this.__currentElement.nodeName === "path") {
            this.__currentElement.setAttribute("paint-order", "stroke fill markers");
        }
        this.__applyCurrentDefaultPath();
        this.__applyStyleToCurrentElement("fill");
    };

    /**
     *  Adds a rectangle to the path.
     */
    ctx.prototype.rect = function(x, y, width, height){
        if(this.__currentElement.nodeName !== "path") {
            this.beginPath();
        }
        this.moveTo(x, y);
        this.lineTo(x+width, y);
        this.lineTo(x+width, y+height);
        this.lineTo(x, y+height);
        this.lineTo(x, y);
        this.closePath();
    };


    /**
     * adds a rectangle element
     */
    ctx.prototype.fillRect = function(x, y, width, height){
        var rect, parent;
        rect = this.__createElement("rect", {
            x : x,
            y : y,
            width : width,
            height : height
        }, true);
        parent = this.__closestGroupOrSvg();
        parent.appendChild(rect);
        this.__currentElement = rect;
        this.__applyStyleToCurrentElement("fill");
    };

    /**
     * Draws a rectangle with no fill
     * @param x
     * @param y
     * @param width
     * @param height
     */
    ctx.prototype.strokeRect = function(x, y, width, height){
        var rect, parent;
        rect = this.__createElement("rect", {
            x : x,
            y : y,
            width : width,
            height : height
        }, true);
        parent = this.__closestGroupOrSvg();
        parent.appendChild(rect);
        this.__currentElement = rect;
        this.__applyStyleToCurrentElement("stroke");
    };


    /**
     * "Clears" a canvas by just drawing a white rectangle in the current group.
     */
    ctx.prototype.clearRect = function(x, y, width, height) {
        var rect, parent = this.__closestGroupOrSvg();
        rect = this.__createElement("rect", {
            x : x,
            y : y,
            width : width,
            height : height,
            fill : "#FFFFFF"
        }, true);
        parent.appendChild(rect);
    };

    /**
     * Adds a linear gradient to a defs tag.
     * Returns a canvas gradient object that has a reference to it's parent def
     */
    ctx.prototype.createLinearGradient = function(x1, y1, x2, y2){
        var grad = this.__createElement("linearGradient", {
            id : randomString(this.__ids),
            x1 : x1+"px",
            x2 : x2+"px",
            y1 : y1+"px",
            y2 : y2+"px",
            "gradientUnits" : "userSpaceOnUse"
        }, false);
        this.__defs.appendChild(grad);
        return new CanvasGradient(grad);
    };

    /**
     * Adds a radial gradient to a defs tag.
     * Returns a canvas gradient object that has a reference to it's parent def
     */
    ctx.prototype.createRadialGradient = function(x0, y0, r0, x1, y1, r1){
        var grad = this.__createElement("radialGradient", {
            id : randomString(this.__ids),
            cx : x1+"px",
            cy : y1+"px",
            r  : r1+"px",
            fx : x0+"px",
            fy : y0+"px",
            "gradientUnits" : "userSpaceOnUse"
        }, false);
        this.__defs.appendChild(grad);
        return new CanvasGradient(grad);

    };

    /**
     * Parses the font string and returns svg mapping
     * @private
     */
    ctx.prototype.__parseFont = function() {
        var regex = /^\s*(?=(?:(?:[-a-z]+\s*){0,2}(italic|oblique))?)(?=(?:(?:[-a-z]+\s*){0,2}(small-caps))?)(?=(?:(?:[-a-z]+\s*){0,2}(bold(?:er)?|lighter|[1-9]00))?)(?:(?:normal|\1|\2|\3)\s*){0,3}((?:xx?-)?(?:small|large)|medium|smaller|larger|[.\d]+(?:\%|in|[cem]m|ex|p[ctx]))(?:\s*\/\s*(normal|[.\d]+(?:\%|in|[cem]m|ex|p[ctx])))?\s*([-,\"\sa-z]+?)\s*$/i;
        var fontPart = regex.exec( this.font );
        var data = {
            style : fontPart[1] || 'normal',
            size : fontPart[4] || '10px',
            family : fontPart[6] || 'sans-serif',
            weight: fontPart[3] || 'normal',
            decoration : fontPart[2] || 'normal',
            href : null
        };

        //canvas doesn't support underline natively, but we can pass this attribute
        if(this.__fontUnderline === "underline") {
            data.decoration = "underline";
        }

        //canvas also doesn't support linking, but we can pass this as well
        if(this.__fontHref) {
            data.href = this.__fontHref;
        }

        return data;
    };

    /**
     * Helper to link text fragments
     * @param font
     * @param element
     * @return {*}
     * @private
     */
    ctx.prototype.__wrapTextLink = function(font, element) {
        if(font.href) {
            var a = this.__createElement("a");
            a.setAttributeNS("http://www.w3.org/1999/xlink", "xlink:href", font.href);
            a.appendChild(element);
            return a;
        }
        return element;
    };

    /**
     * Fills or strokes text
     * @param text
     * @param x
     * @param y
     * @param action - stroke or fill
     * @private
     */
    ctx.prototype.__applyText = function(text, x, y, action) {
        var font = this.__parseFont(),
            parent = this.__closestGroupOrSvg(),
            textElement = this.__createElement("text", {
                "font-family" : font.family,
                "font-size" : font.size,
                "font-style" : font.style,
                "font-weight" : font.weight,
                "text-decoration" : font.decoration,
                "x" : x,
                "y" : y,
                "text-anchor": getTextAnchor(this.textAlign),
                "dominant-baseline": getDominantBaseline(this.textBaseline)
            }, true);

        textElement.appendChild(document.createTextNode(text));
        this.__currentElement = textElement;
        this.__applyStyleToCurrentElement(action);
        parent.appendChild(this.__wrapTextLink(font,textElement));
    };

    /**
     * Creates a text element
     * @param text
     * @param x
     * @param y
     */
    ctx.prototype.fillText = function(text, x, y){
        this.__applyText(text, x, y, "fill");
    };

    /**
     * Strokes text
     * @param text
     * @param x
     * @param y
     */
    ctx.prototype.strokeText = function(text, x, y){
        this.__applyText(text, x, y, "stroke");
    };

    /**
     * No need to implement this for svg.
     * @param text
     * @return {TextMetrics}
     */
    ctx.prototype.measureText = function(text){
        this.__ctx.font = this.font;
        return this.__ctx.measureText(text);
    };

    /**
     *  Arc command!
     */
    ctx.prototype.arc = function(x, y, radius, startAngle, endAngle, counterClockwise) {
        startAngle = startAngle % (2*Math.PI);
        endAngle = endAngle % (2*Math.PI);
        if(startAngle === endAngle) {
            //circle time! subtract some of the angle so svg is happy (svg elliptical arc can't draw a full circle)
            endAngle = ((endAngle + (2*Math.PI)) - 0.001 * (counterClockwise ? -1 : 1)) % (2*Math.PI);
        }
        var endX = x+radius*Math.cos(endAngle),
            endY = y+radius*Math.sin(endAngle),
            startX = x+radius*Math.cos(startAngle),
            startY = y+radius*Math.sin(startAngle),
            sweepFlag = counterClockwise ? 0 : 1,
            largeArcFlag = 0,
            diff = endAngle - startAngle;

        // https://github.com/gliffy/canvas2svg/issues/4
        if(diff < 0) {
            diff += 2*Math.PI;
        }

        if(counterClockwise) {
            largeArcFlag = diff > Math.PI ? 0 : 1;
        } else {
            largeArcFlag = diff > Math.PI ? 1 : 0;
        }

        this.lineTo(startX, startY);
        this.__addPathCommand(format("A {rx} {ry} {xAxisRotation} {largeArcFlag} {sweepFlag} {endX} {endY}",
            {rx:radius, ry:radius, xAxisRotation:0, largeArcFlag:largeArcFlag, sweepFlag:sweepFlag, endX:endX, endY:endY}));

        this.__currentPosition = {x: endX, y: endY};
    };

    /**
     * Generates a ClipPath from the clip command.
     */
    ctx.prototype.clip = function(){
        var group = this.__closestGroupOrSvg(),
            clipPath = this.__createElement("clipPath"),
            id =  randomString(this.__ids),
            newGroup = this.__createElement("g");

        group.removeChild(this.__currentElement);
        clipPath.setAttribute("id", id);
        clipPath.appendChild(this.__currentElement);

        this.__defs.appendChild(clipPath);

        //set the clip path to this group
        group.setAttribute("clip-path", format("url(#{id})", {id:id}));

        //clip paths can be scaled and transformed, we need to add another wrapper group to avoid later transformations
        // to this path
        group.appendChild(newGroup);

        this.__currentElement = newGroup;

    };

    /**
     * Draws a canvas, image or mock context to this canvas.
     * Note that all svg dom manipulation uses node.childNodes rather than node.children for IE support.
     * http://www.whatwg.org/specs/web-apps/current-work/multipage/the-canvas-element.html#dom-context-2d-drawimage
     */
    ctx.prototype.drawImage = function(){
        //convert arguments to a real array
        var args = Array.prototype.slice.call(arguments),
            image=args[0],
            dx, dy, dw, dh, sx=0, sy=0, sw, sh, parent, svg, defs, group,
            currentElement, svgImage, canvas, context, id;

        if(args.length === 3) {
            dx = args[1];
            dy = args[2];
            sw = image.width;
            sh = image.height;
            dw = sw;
            dh = sh;
        } else if(args.length === 5) {
            dx = args[1];
            dy = args[2];
            dw = args[3];
            dh = args[4];
            sw = image.width;
            sh = image.height;
        } else if(args.length === 9) {
            sx = args[1];
            sy = args[2];
            sw = args[3];
            sh = args[4];
            dx = args[5];
            dy = args[6];
            dw = args[7];
            dh = args[8];
        } else {
            throw new Error("Inavlid number of arguments passed to drawImage: " + arguments.length);
        }

        parent = this.__closestGroupOrSvg();
        currentElement = this.__currentElement;

        if(image instanceof ctx) {
            //canvas2svg mock canvas context. In the future we may want to clone nodes instead.
            //also I'm currently ignoring dw, dh, sw, sh, sx, sy for a mock context.
            svg = image.getSvg();
            defs = svg.childNodes[0];
            while(defs.childNodes.length) {
                id = defs.childNodes[0].getAttribute("id");
                this.__ids[id] = id;
                this.__defs.appendChild(defs.childNodes[0]);
            }
            group = svg.childNodes[1];
            parent.appendChild(group);
            this.__currentElement = group;
            this.translate(dx, dy);
            this.__currentElement = currentElement;
        } else if(image.nodeName === "CANVAS" || image.nodeName === "IMG") {
            //canvas or image
            svgImage = this.__createElement("image");
            svgImage.setAttribute("width", dw);
            svgImage.setAttribute("height", dh);
            svgImage.setAttribute("preserveAspectRatio", "none");

            if(sx || sy || sw !== image.width || sh !== image.height) {
                //crop the image using a temporary canvas
                canvas = document.createElement("canvas");
                canvas.width = dw;
                canvas.height = dh;
                context = canvas.getContext("2d");
                context.drawImage(image, sx, sy, sw, sh, 0, 0, dw, dh);
                image = canvas;
            }

            svgImage.setAttributeNS("http://www.w3.org/1999/xlink", "xlink:href",
                image.nodeName === "CANVAS" ? image.toDataURL() : image.getAttribute("src"));
            parent.appendChild(svgImage);
            this.__currentElement = svgImage;
            this.translate(dx, dy);
            this.__currentElement = currentElement;
        }
    };

    /**
     * Generates a pattern tag
     */
    ctx.prototype.createPattern = function(image, repetition){
        var pattern = document.createElementNS("http://www.w3.org/2000/svg", "pattern"), id = randomString(this.__ids),
            img;
        pattern.setAttribute("id", id);
        pattern.setAttribute("width", image.width);
        pattern.setAttribute("height", image.height);
        if(image.nodeName === "CANVAS" || image.nodeName === "IMG") {
            img = document.createElementNS("http://www.w3.org/2000/svg", "image");
            img.setAttribute("width", image.width);
            img.setAttribute("height", image.height);
            img.setAttributeNS("http://www.w3.org/1999/xlink", "xlink:href",
                image.nodeName === "CANVAS" ? image.toDataURL() : image.getAttribute("src"));
            pattern.appendChild(img);
            this.__defs.appendChild(pattern);
        } else if(image instanceof ctx) {
            pattern.appendChild(image.__root.childNodes[1]);
            this.__defs.appendChild(pattern);
        }
        return new CanvasPattern(pattern, this);
    };

    /**
     * Not yet implemented
     */
    ctx.prototype.drawFocusRing = function(){};
    ctx.prototype.createImageData = function(){};
    ctx.prototype.getImageData = function(){};
    ctx.prototype.putImageData = function(){};
    ctx.prototype.globalCompositeOperation = function(){};
    ctx.prototype.setTransform = function(){};

    //add options for alternative namespace
    module.exports = ctx;

}());

},{}],2:[function(require,module,exports){
var C2S = require('./canvas2svg');

var Context = function(width, height, options) {
    C2S.call(this);
    this.__width = width;
    this.__height = height;
    this.generations = [[]]; // used to collect element references for different generations

    var _this = this;
    this.__imageSmoothingEnabled = true;

    ["mozImageSmoothingEnabled",
     "webkitImageSmoothingEnabled",
     "msImageSmoothingEnabled",
     "imageSmoothingEnabled"].forEach(function(k) {
         Object.defineProperty(_this, k, {
             get: function() {
                 return _this.__imageSmoothingEnabled;
             },
             set: function(val) {
                 _this.__imageSmoothingEnabled = val;
             }
         });
     });

    options = options || {};

    ["fillStyle", "strokeStyle"].forEach(function(prop) {
        var key = "__" + prop;
        Object.defineProperty(_this, prop, {
            get: function() {
                return _this[key];
            },
            set: function(val) {
                if (val.indexOf('NaN') > -1) {
                    console.warn("svgcanvas: invalid value for " + prop + ", fail to set it to " + val);
                    return;
                }
                _this[key] = val;
            }
        });
    });


    if (options.debug) {
        this.__history = []; // method history

        var methods = [];
        for(var key in this) {
            if (typeof this[key] === "function") {
                if (key.indexOf('__') !== 0) {
                    if (key !== 'getSerializedSvg') {
                        methods.push(key);
                    }
                }
            }
        }
        ["__fillStyle", "__strokeStyle"].forEach(function(prop) {
            var key = "__debug__" + prop;
            Object.defineProperty(_this, prop, {
                get: function() {
                    return _this[key];
                },
                set: function(val) {
                    var call = prop.replace(/__/g, '') + " = " + val;
                    _this.__history.push(call);
                    _this[key] = val;
                }
            });
        });
        methods.forEach(function(method) {
            var fn = _this[method];
            _this[method] = function() {
                var call = method + '(' + Array.prototype.slice.call(arguments).join(', ') + ');';

                // keep call history
                _this.__history.push(call);
                if (_this.__history.length > 100) {
                    _this.__history.shift();
                }

                return fn.apply(_this, arguments);
            };
        });
    }
};

Context.prototype = Object.create(C2S.prototype);

Context.prototype.scale = function(x, y) {
    if (x === undefined || y === undefined) {
        return;
    } else {
        C2S.prototype.scale.apply(this, arguments);
    }
};

Context.prototype.__createElement = function(elementName, properties, resetFill) {
    if (!this.__imageSmoothingEnabled) {
        // only shape elements can use the shape-rendering attribute
        if (["circle", "ellipse", "line", "path", "polygon", "polyline", "rect"].indexOf(elementName) > -1) {
            properties = properties || {};
            properties["shape-rendering"] = "crispEdges"; // disable anti-aliasing
        }
    }

    var element = C2S.prototype.__createElement.call(this, elementName, properties, resetFill);
    var currentGeneration = this.generations[this.generations.length - 1];
    currentGeneration.push(element);
    return element;
};

Context.prototype.__gc = function() {
    this.generations.push([]);
    var ctx = this;
    // make sure it happens after current job done
    // for example: in p5.js's redraw use setTimeout will make gc called after both save() and restore() called
    setTimeout(function() {
        if (ctx.__groupStack.length > 0) {
            // we are between ctx.save() and ctx.restore(), skip gc
            return;
        }
        if (ctx.__currentElement.nodeName === 'path') {
            // we are still in path, skip gc
            return;
        }
        // keep only latest generation
        while (ctx.generations.length > 1) {
            var elements = ctx.generations.shift();
            var lastCount = 0;
            var count = elements.length;
            while (count > 0) {
                lastCount = count;
                elements = elements.filter(function(elem) {
                    // in case children may from live generation, gc from bottom to top
                    var children = elem.children || elem.childNodes; // childNodes for IE
                    if (children.length === 0) {
                        elem.parentNode.removeChild(elem);
                        return false;
                    } else {
                        return true;
                    }
                });
                count = elements.length;
                if (count === lastCount) {
                    // could not gc more, exit now
                    // save this elements to live generation
                    var liveGeneration = ctx.generations[ctx.generations.length - 1];
                    elements.forEach(function(elem) {
                        liveGeneration.push(elem);
                    });
                    // exit
                    break;
                }
            }
        }
    }, 0);
};

/**
 * Clear full canvas and do gc
 * @private
 */
Context.prototype.__clearCanvas = function() {
    // remove all
    this.generations.forEach(function(elems) {
        elems.forEach(function(elem) {
            if (elem) {
                elem.parentNode.removeChild(elem);
            }
        });
    });
    this.generations = [[]];
    var g = this.__createElement('g');
    this.__root.appendChild(g);
    this.__currentElement = g;
};

Context.prototype.clearRect = function(x, y, w, h) {
    if (x === 0 && y === 0 && w === this.__width && h === this.__height) {
        this.__clearCanvas();
    } else {
        C2S.prototype.clearRect.call(this, x, y, w, h);
    }
};

Context.prototype.fillRect = function(x, y, w, h) {
    if (x === 0 && y === 0 && w === this.__width && h === this.__height) {
        this.__gc();
    }
    C2S.prototype.fillRect.call(this, x, y, w, h);
};

// Simple version of drawImage
// Note that this version does not handle drawing mock context
Context.prototype.drawImage = function() {
    var canvas = document.createElement('canvas');
    canvas.width = this.__width;
    canvas.height = this.__height;
    var args = arguments;
    var ctx = canvas.getContext('2d');
    ctx.drawImage.apply(ctx, args);
    // Note: don't use foreign object,
    // otherwise the saved SVG may be unusable for other application
    var url = canvas.toDataURL('image/png');
    var image = this.__createElement('image', {
        x: 0,
        y: 0,
        width: canvas.width,
        height: canvas.height,
        preserveAspectRatio: 'none'
    });
    var parent = this.__closestGroupOrSvg();
    image.setAttributeNS("http://www.w3.org/1999/xlink", "xlink:href", url);
    parent.appendChild(image);
};

Context.prototype.getSerializedSvg = null;

module.exports = Context;

},{"./canvas2svg":1}],3:[function(require,module,exports){
var Context = require('./context');

function SVGCanvas(options) {

    var debug = options && options.debug;

    this.ctx = new Context(100, 100, {debug: debug});
    this.svg = this.ctx.__root;

    // sync attributes to svg
    var svg = this.svg;
    var _this = this;

    var wrapper = document.createElement('div');
    wrapper.style.display = 'inline-block';
    wrapper.appendChild(svg);
    this.wrapper = wrapper;

    Object.defineProperty(this, 'className', {
        get: function() {
            return wrapper.getAttribute('class') || '';
        },
        set: function(val) {
            return wrapper.setAttribute('class', val);
        }
    });

    ["width", "height"].forEach(function(prop) {
        Object.defineProperty(_this, prop, {
            get: function() {
                return svg.getAttribute(prop) | 0;
            },
            set: function(val) {
                if (isNaN(val) || (typeof val === "undefined")) {
                    return;
                }
                _this.ctx['__'+prop] = val;
                svg.setAttribute(prop, val);
                return wrapper[prop] = val;
            }
        });
    });

    ["style", "id"].forEach(function(prop) {
        Object.defineProperty(_this, prop, {
            get: function() {
                return wrapper[prop];
            },
            set: function(val) {
                if (typeof val !== "undefined") {
                    return wrapper[prop] = val;
                }
            }
        });
    });

    ["getBoundingClientRect"].forEach(function(fn) {
        _this[fn] = function() {
            return svg[fn]();
        };
    });
}

SVGCanvas.prototype.getContext = function(type) {
    if (type !== '2d') {
        throw new Error('Unsupported type of context for SVGCanvas');
    }

    return this.ctx;
};

// you should always use URL.revokeObjectURL after your work done
SVGCanvas.prototype.toObjectURL = function() {
    var data = new XMLSerializer().serializeToString(this.svg);
    var svg = new Blob([data], {type: 'image/svg+xml;charset=utf-8'});
    return URL.createObjectURL(svg);
};

SVGCanvas.prototype.toDataURL = function(type, options) {
    var xml = new XMLSerializer().serializeToString(this.svg);

    // documentMode is an IE-only property
    // http://msdn.microsoft.com/en-us/library/ie/cc196988(v=vs.85).aspx
    // http://stackoverflow.com/questions/10964966/detect-ie-version-prior-to-v9-in-javascript
    var isIE = document.documentMode;

    if (isIE) {
        // This is patch from canvas2svg
        // IE search for a duplicate xmnls because they didn't implement setAttributeNS correctly
        var xmlns = /xmlns="http:\/\/www\.w3\.org\/2000\/svg".+xmlns="http:\/\/www\.w3\.org\/2000\/svg/gi;
        if(xmlns.test(xml)) {
            xml = xml.replace('xmlns="http://www.w3.org/2000/svg','xmlns:xlink="http://www.w3.org/1999/xlink');
        }
    }

    var SVGDataURL = "data:image/svg+xml;charset=utf-8," + encodeURIComponent(xml);
    if (type === "image/svg+xml" || !type) {
        return SVGDataURL;
    }
    if (type === "image/jpeg" || type === "image/png") {
        var canvas = document.createElement('canvas');
        canvas.width = this.width;
        canvas.height = this.height;
        var ctx = canvas.getContext('2d');
        var img = new Image();
        img.src = SVGDataURL;
        if (img.complete && img.width > 0 && img.height > 0) {
            // for chrome, it's ready immediately
            ctx.drawImage(img, 0, 0);
            return canvas.toDataURL(type, options);
        } else {
            // for firefox, it's not possible to provide sync api in current thread
            // and web worker doesn't provide canvas API, so
            throw new Error('svgcanvas.toDataURL() for jpeg/png is only available in Chrome.');
        }
    }
    throw new Error('Unknown type for SVGCanvas.prototype.toDataURL, please use image/jpeg | image/png | image/svg+xml.');
};

SVGCanvas.prototype.addEventListener = function() {
    return this.svg.addEventListener.apply(this, arguments);
};

// will return wrapper element: <div><svg></svg></div>
SVGCanvas.prototype.getElement = function() {
    return this.wrapper;
};

module.exports = SVGCanvas;

},{"./context":2}],4:[function(require,module,exports){
var constants = {
    SVG: 'svg'
};

module.exports = constants;

},{}],5:[function(require,module,exports){
module.exports = function(p5) {
    /**
     * Returns an Array of SVGElements of current SVG Graphics matching given selector
     *
     * @function querySVG
     * @memberof p5.prototype
     * @param {String} selector CSS selector for query
     * @returns {SVGElement[]}
     */
    p5.prototype.querySVG = function(selector) {
        var svg = this._renderer && this._renderer.svg;
        if (!svg) {
            return null;
        }
        return p5.SVGElement.prototype.query.call({elt: svg}, selector);
    };

    /**
     * @namespace SVGElement
     * @constructor
     * @param {Element} element
     */
    function SVGElement(element) {
        if (!element) {
            return null;
        }
        return p5.Element.apply(this, arguments);
    }

    SVGElement.prototype = Object.create(p5.Element.prototype);

    /**
     * Returns an Array of children of current SVG Element matching given selector
     *
     * @function query
     * @memberof SVGElement.prototype
     * @param {String} selector CSS selector for query
     * @returns {SVGElement[]}
     */
    SVGElement.prototype.query = function(selector) {
        var elements = this.elt.querySelectorAll(selector);
        var objects = [];
        for (var i = 0; i < elements.length; i++) {
            objects[i] = new SVGElement(elements[i]);
        }
        return objects;
    };

    /**
     * Append a new child to current element.
     *
     * @function append
     * @memberof SVGElement.prototype
     * @param {SVGElement|Element} element
     */
    SVGElement.prototype.append = function(element) {
        var elt = element.elt || element;
        this.elt.appendChild(elt);
        return this;
    };

    /**
     * Apply different attribute operation based on arguments.length
     * <ul>
     *     <li>setAttribute(name, value)</li>
     *     <li>setAttributeNS(namespace, name, value)</li>
     *     <li>getAttribute(name)</li>
     * </ul>
     *
     * @function attribute
     * @memberof SVGElement.prototype
     */
    SVGElement.prototype.attribute = function() {
        var args = arguments;
        if (args.length === 3) {
            this.elt.setAttributeNS.apply(this.elt, args);
        }
        if (args.length === 2) {
            this.elt.setAttribute.apply(this.elt, args);
        }
        if (args.length === 1) {
            return this.elt.getAttribute.apply(this.elt, args);
        }
        return this;
    };

    /**
     * Apply filter on current element.
     * If called multiple times,
     * these filters will be chained together and combine to a bigger SVG filter.
     *
     * @function filter
     * @memberof SVGElement.prototype
     * @param {String} filter BLUR, GRAY, INVERT, THRESHOLD, OPAQUE, ERODE, DILATE (defined in p5's constants)
     * @param {Any} argument Argument for that filter
     */
    SVGElement.prototype.filter = function(filter, arg) {
        p5.SVGFilters.apply(this, filter, arg);
        return this;
    };

    /**
     * Remove applied filter on current element
     * After called, rest filters will be chained together
     * and combine to a new SVG filter.
     *
     * @function unfilter
     * @memberof SVGElement.prototype
     * @param {String} filter BLUR, GRAY, INVERT, THRESHOLD, OPAQUE, ERODE, DILATE (defined in p5's constants)
     * @param {Any} argument Argument for that filter
     */
    SVGElement.prototype.unfilter = function(filterName, arg) {
        var filters = this.attribute('data-p5-svg-filters') || '[]';
        filters = JSON.parse(filters);
        if (arg === undefined) {
            arg = null;
        }
        var found = false;
        filters = filters.reverse().filter(function(filter) {
            if ((filter[0] === filterName) && (filter[1] === arg) && !found) {
                found = true;
                return false;
            }
            return true;
        }).reverse();
        this.attribute('data-p5-svg-filters', JSON.stringify(filters));
        p5.SVGFilters.apply(this, null);
        return this;
    };

    /**
     * Create SVGElement
     *
     * @function create
     * @memberof SVGElement
     * @param {String} nodeName
     * @param {Object} [attributes] Attributes for the element
     * @return {SVGElement}
     */
    SVGElement.create = function(nodeName, attributes) {
        attributes = attributes || {};
        var elt = document.createElementNS('http://www.w3.org/2000/svg', nodeName);
        Object.keys(attributes).forEach(function(k) {
            elt.setAttribute(k, attributes[k]);
        });
        return new SVGElement(elt);
    };

    /**
     * Tell if current element matching given selector.
     * This is polyfill from MDN.
     *
     * @see https://developer.mozilla.org/en-US/docs/Web/API/Element/matches
     *
     * @function matches
     * @memberof SVGElement.prototype
     * @param {String} selector CSS Selector
     * @return {Bool}
     */
    SVGElement.prototype.matches = function(selector) {
        var element = this.elt;
        var matches = (element.document || element.ownerDocument).querySelectorAll(selector);
        var i = 0;
        while (matches[i] && matches[i] !== element) {
            i++;
        }
        return matches[i] ? true : false;
    };

    /**
     * Get defs element, or create one if not exists
     *
     * @private
     */
    SVGElement.prototype._getDefs = function() {
        var svg = this.parentNode('svg');
        var defs = svg.query('defs');
        if (defs[0]) {
            defs = defs[0];
        } else {
            defs = SVGElement.create('defs');
            svg.append(defs);
        }
        return defs;
    };

    /**
     * Get parentNode.
     * If selector not given, returns parentNode.
     * Otherwise, will look up all ancestors,
     * and return closest element matching given selector,
     * or return null if not found.
     *
     * @function parentNode
     * @memberof SVGElement.prototype
     * @param {String} [selector] CSS Selector
     * @return {SVGElement}
     */
    SVGElement.prototype.parentNode = function(selector) {
        if (!selector) {
            return new SVGElement(this.elt.parentNode);
        }
        var elt = this;
        while (elt) {
            elt = this.parentNode();
            if (elt && elt.matches(selector)) {
                return elt;
            }
        }
        return null;
    };

    p5.SVGElement = SVGElement;
};

},{}],6:[function(require,module,exports){
// SVG Filter

module.exports = function(p5) {
    var _filter = p5.prototype.filter;

    var SVGFilters = require('./p5.SVGFilters')(p5);

    /**
     * Register a custom SVG Filter
     *
     * @function registerSVGFilter
     * @memberof p5.prototype
     * @param {String} name Name for Custom SVG filter
     * @param {Function} filterFunction filterFunction(inGraphicsName, resultGraphicsName, value)
     *                                  should return SVGElement or Array of SVGElement.
     * @example
     * registerSVGFilter('myblur', function(inGraphicsName, resultGraphicsName, value) {
     *     return SVGElement.create('feGaussianBlur', {
     *         stdDeviation: val,
     *         in: inGraphics,
     *         result: resultGraphics,
     *         'color-interpolation-filters': 'sRGB'
     *     });
     * });
     * filter('myblur', 5);
     */
    p5.prototype.registerSVGFilter = function(name, fn) {
        SVGFilters[name] = fn;
    };

    p5.prototype.filter = function(operation, value) {
        var svg = this._renderer.svg;
        if (svg) {
            // move nodes to a new <g>
            var nodes = svg.children || svg.childNodes; // childNodes is for IE
            var g = p5.SVGElement.create('g');
            this._renderer._setGCFlag(g.elt);
            svg.appendChild(g.elt);
            // convert nodeList to array and use forEach
            // instead of using for loop,
            // which is buggy due to the length changed during append
            nodes = Array.prototype.slice.call(nodes);
            nodes.forEach(function(node) {
                if (node !== g.elt && (node.nodeName.toLowerCase() !== 'defs')) {
                    g.elt.appendChild(node);
                }
            });

            // apply filter
            g.filter(operation, value);

            // create new <g> so that new element won't be influenced by the filter
            g = p5.SVGElement.create('g');
            this._renderer._setGCFlag(g.elt);
            this._renderer.svg.appendChild(g.elt);
            this._renderer.drawingContext.__currentElement = g.elt;
        } else {
            _filter.apply(this, arguments);
        }
    };
};

},{"./p5.SVGFilters":10}],7:[function(require,module,exports){
module.exports = function(p5) {
    /**
     * @namespace p5
     */
    require('./p5.RendererSVG')(p5);
    require('./rendering')(p5);
    require('./io')(p5);
    require('./element')(p5);
    require('./filters')(p5);

    // attach constants to p5 instance
    var constants = require('./constants');
    Object.keys(constants).forEach(function(k) {
        p5.prototype[k] = constants[k];
    });
};

},{"./constants":4,"./element":5,"./filters":6,"./io":8,"./p5.RendererSVG":9,"./rendering":11}],8:[function(require,module,exports){
module.exports = function(p5) {
    /**
     * Convert SVG Element to jpeg / png data url
     *
     * @private
     * @param {SVGElement} svg SVG Element
     * @param {String} mine Mine
     * @param {Function} callback
     */
    var svg2img = function(svg, mine, callback) {
        svg = (new XMLSerializer()).serializeToString(svg);
        svg = 'data:image/svg+xml;charset=utf-8,' + encodeURI(svg);
        if (mine == 'image/svg+xml') {
            callback(null, svg);
            return;
        }
        var img = new Image();
        var canvas = document.createElement('canvas');
        var ctx = canvas.getContext('2d');
        img.onload = function() {
            canvas.width = img.width;
            canvas.height = img.height;
            ctx.drawImage(img, 0, 0);
            var dataURL = canvas.toDataURL(mine);
            callback(null, dataURL);
        };
        img.src = svg;
    };

    /**
     * Get SVG frame, and convert to target type
     *
     * @private
     * @param {Object} options
     * @param {SVGElement} options.svg SVG Element, defaults to current svg element
     * @param {String} options.filename
     * @param {String} options.ext Extension: 'svg' or 'jpg' or 'jpeg' or 'png'
     * @param {Function} options.callback
     */
    p5.prototype._makeSVGFrame = function(options) {
        var filename = options.filename || 'untitled';
        var ext = options.extension;
        ext = ext || this._checkFileExtension(filename, ext)[1];
        var regexp = new RegExp('\\.' + ext + '$');
        filename = filename.replace(regexp, '');
        if (ext === '') {
            ext = 'svg';
        }
        var mine = {
            png: 'image/png',
            jpeg: 'image/jpeg',
            jpg: 'image/jpeg',
            svg: 'image/svg+xml'
        }[ext];
        if (!mine) {
            throw new Error('Fail to getFrame, invalid extension: ' + ext + ', please use png | jpeg | jpg | svg.');
        }

        var svg = options.svg || this._renderer.svg;
        svg2img(svg, mine, function(err, dataURL) {
            var downloadMime = 'image/octet-stream';
            dataURL = dataURL.replace(mine, downloadMime);
            options.callback(err, {
                imageData: dataURL,
                filename: filename,
                ext: ext
            });
        });
    };

    /**
     * Save the current SVG as an image. In Safari, will open the
     * image in the window and the user must provide their own
     * filename on save-as. Other browsers will either save the
     * file immediately, or prompt the user with a dialogue window.
     *
     * @function saveSVG
     * @memberof p5.prototype
     * @param {Graphics|Element|SVGElement} [svg] Source to save
     * @param {String} [filename]
     * @param {String} [extension] Extension: 'svg' or 'jpg' or 'jpeg' or 'png'
     */
    p5.prototype.saveSVG = function() {
        // don't use slice on arguments because it prevents optimizations
        var args = arguments;
        args = [args[0], args[1], args[2]];

        var svg;

        if (args[0] instanceof p5.Graphics) {
            svg = args[0]._renderer.svg;
            args.shift();
        }

        if (args[0] && args[0].elt) {
            svg = args[0].elt;
            args.shift();
        }

        if (typeof args[0] == 'object') {
            svg = args[0];
            args.shift();
        }

        var filename = args[0];
        var ext = args[1];

        var p = this;
        this._makeSVGFrame({
            svg: svg,
            filename: filename,
            extension: ext,
            callback: function(err, frame) {
                p.downloadFile(frame.imageData, frame.filename, frame.ext);
            }
        });
    };

    /**
     * Extends p5's saveFrames with SVG support
     *
     * @function saveFrames
     * @memberof p5.prototype
     * @param {String} filename filename
     * @param {String} extension Extension: 'svg' or 'jpg' or 'jpeg' or 'png'
     * @param {Number} duration duration
     * @param {Number} fps fps
     * @param {Function} callback callback
     */
    var _saveFrames = p5.prototype.saveFrames;
    p5.prototype.saveFrames = function(filename, extension, duration, fps, callback) {
        var args = arguments;

        if (!this._renderer.svg) {
            _saveFrames.apply(this, args);
            return;
        }

        duration = duration || 3;
        duration = p5.prototype.constrain(duration, 0, 15);
        duration = duration * 1000;
        fps = fps || 15;
        fps = p5.prototype.constrain(fps, 0, 22);
        var count = 0;

        var frames = [];
        var pending = 0;

        var p = this;
        var frameFactory = setInterval(function () {
            (function(count) {
                pending++;
                p._makeSVGFrame({
                    filename: filename + count,
                    extension: extension,
                    callback: function(err, frame) {
                        frames[count] = frame;
                        pending--;
                    }
                });
            })(count);
            count++;
        }, 1000 / fps);

        var done = function() {
            if (pending > 0) {
                setTimeout(function() {
                    done();
                }, 10);
                return;
            }
            if (callback) {
                callback(frames);
            } else {
                frames.forEach(function(f) {
                    p.downloadFile(f.imageData, f.filename, f.ext);
                });
            }
        };

        setTimeout(function () {
            clearInterval(frameFactory);
            done();
        }, duration + 0.01);
    };

    /**
     * Extends p5's save method with SVG support
     *
     * @function save
     * @memberof p5.prototype
     * @param {Graphics|Element|SVGElement} [source] Source to save
     * @param {String} [filename] filename
     */
    var _save = p5.prototype.save;
    p5.prototype.save = function() {
        var args = arguments;
        args = [args[0], args[1]];

        var svg;

        if (args[0] instanceof p5.Graphics) {
            var svgcanvas = args[0].elt;
            svg = svgcanvas.svg;
            args.shift();
        }

        if (args[0] && args[0].elt) {
            svg = args[0].elt;
            args.shift();
        }

        if (typeof args[0] == 'object') {
            svg = args[0];
            args.shift();
        }

        svg = svg || (this._renderer && this._renderer.svg);

        var filename = args[0];
        var supportedExtensions = ['jpeg', 'png', 'jpg', 'svg', ''];
        var ext = this._checkFileExtension(filename, '')[1];

        var useSVG = svg && svg.nodeName && svg.nodeName.toLowerCase() === 'svg' && supportedExtensions.indexOf(ext) > -1;

        if (useSVG) {
            this.saveSVG(svg, filename);
        } else {
            return _save.apply(this, arguments);
        }
    };

    /**
     * Custom get in p5.svg (handles http and dataurl)
     * @private
     */
    p5.prototype._svg_get = function(path, successCallback, failureCallback) {
        if (path.indexOf('data:') === 0) {
            if (path.indexOf(',') === -1) {
                failureCallback(new Error('Fail to parse dataurl: ' + path));
                return;
            }
            var svg = path.split(',').pop();
            // force request to dataurl to be async
            // so that it won't make preload mess
            setTimeout(function() {
                if (path.indexOf(';base64,') > -1) {
                    svg = atob(svg);
                } else {
                    svg = decodeURIComponent(svg);
                }
                successCallback(svg);
            }, 1);
            return svg;
        } else {
            this.httpGet(path, successCallback);
            return null;
        }
    };

    /**
     * loadSVG (like loadImage, but will return SVGElement)
     *
     * @function loadSVG
     * @memberof p5.prototype
     * @returns {p5.SVGElement}
     */
    p5.prototype.loadSVG = function(path, successCallback, failureCallback) {
        var div = document.createElement('div');
        var element = new p5.SVGElement(div);
        this._svg_get(path, function(svg) {
            div.innerHTML = svg;
            svg = div.querySelector('svg');
            if (!svg) {
                if (failureCallback) {
                    failureCallback(new Error('Fail to create <svg>.'));
                }
                return;
            }
            element.elt = svg;
            if (successCallback) {
                successCallback(element);
            }
        }, failureCallback);
        return element;
    };
    // cause preload to wait
    p5.prototype._preloadMethods.loadSVG = p5.prototype;

    p5.prototype.getDataURL = function() {
        return this._renderer.elt.toDataURL('image/svg+xml');
    };
};

},{}],9:[function(require,module,exports){
var SVGCanvas = require('svgcanvas');

module.exports = function(p5) {
    /**
     * @namespace RendererSVG
     * @constructor
     * @param {Element} elt canvas element to be replaced
     * @param {p5} pInst p5 Instance
     * @param {Bool} isMainCanvas
     */
    function RendererSVG(elt, pInst, isMainCanvas) {
        var svgCanvas = new SVGCanvas();
        var svg = svgCanvas.svg;

        // replace <canvas> with <svg> and copy id, className
        var parent = elt.parentNode;
        var id = elt.id;
        var className = elt.className;
        parent.replaceChild(svgCanvas.getElement(), elt);
        svgCanvas.id = id;
        svgCanvas.className = className;
        elt = svgCanvas; // our fake <canvas>

        elt.parentNode = {
            // fake parentNode.removeChild so that noCanvas will work
            removeChild: function(element) {
                if (element === elt) {
                    var wrapper = svgCanvas.getElement();
                    wrapper.parentNode.removeChild(wrapper);
                }
            }
        };

        p5.Renderer2D.call(this, elt, pInst, isMainCanvas);

        this.isSVG = true;
        this.svg = svg;

        return this;
    }

    RendererSVG.prototype = Object.create(p5.Renderer2D.prototype);

    RendererSVG.prototype._applyDefaults = function() {
        p5.Renderer2D.prototype._applyDefaults.call(this);
        this.drawingContext.lineWidth = 1;
    };

    RendererSVG.prototype.line = function(x1, y1, x2, y2) {
        var styleEmpty = 'rgba(0,0,0,0)';
        var ctx = this.drawingContext;
        if (!this._doStroke) {
            return this;
        } else if(ctx.strokeStyle === styleEmpty){
            return this;
        }
        ctx.beginPath();
        ctx.moveTo(x1, y1);
        ctx.lineTo(x2, y2);
        ctx.stroke();
        return this;
    };

    RendererSVG.prototype.resize = function(w, h) {

        // console.log({w: w, h: h, tw: this.width, th: this.height});

        if (!w || !h) {
            // ignore invalid values for width and height
            return;
        }
        if (this.width !== w || this.height !== h) {
            // canvas will be cleared if its size changed
            // so, we do same thing for SVG
            // note that at first this.width and this.height is undefined
            this.drawingContext.__clearCanvas();
        }
        this._withPixelDensity(function() {
            p5.Renderer2D.prototype.resize.call(this, w, h);
        });
        // For scale, crop
        // see also: http://sarasoueidan.com/blog/svg-coordinate-systems/
        this.svg.setAttribute('viewBox', [0, 0, w, h].join(' '));
    };

    /**
     * @private
     */
    RendererSVG.prototype._withPixelDensity = function(fn) {
        var pixelDensity = this._pInst._pixelDensity;
        this._pInst._pixelDensity = 1; // 1 is OK for SVG
        fn.apply(this);
        this._pInst._pixelDensity = pixelDensity;
    };

    RendererSVG.prototype.background = function() {
        var args = arguments;
        this._withPixelDensity(function() {
            p5.Renderer2D.prototype.background.apply(this, args);
        });
    };

    RendererSVG.prototype.resetMatrix = function() {
        this._withPixelDensity(function() {
            p5.Renderer2D.prototype.resetMatrix.apply(this);
        });
    };

    /**
     * set gc flag for svgcanvas
     *
     * @private
     */
    RendererSVG.prototype._setGCFlag = function(element) {
        var that = this.drawingContext;
        var currentGeneration = that.generations[that.generations.length - 1];
        currentGeneration.push(element);
    };

    /**
     * Append a element to current SVG Graphics
     *
     * @function appendChild
     * @memberof RendererSVG.prototype
     * @param {SVGElement|Element} element
     */
    RendererSVG.prototype.appendChild = function(element) {
        if (element && element.elt) {
            element = element.elt;
        }
        this._setGCFlag(element);
        var g = this.drawingContext.__closestGroupOrSvg();
        g.appendChild(element);
    };

    /**
     * Draw an image or SVG to current SVG Graphics
     *
     * FIXME: sx, sy, sWidth, sHeight
     *
     * @function image
     * @memberof RendererSVG.prototype
     * @param {p5.Graphics|SVGGraphics|SVGElement|Element} image
     * @param {Number} x
     * @param {Number} y
     * @param {Number} width
     * @param {Number} height
     */
    RendererSVG.prototype.image = function(img,  sx, sy, sWidth, sHeight, x, y, w, h) {
        if (!img) {
            throw new Error('Invalid image: ' + img);
        }
        var elt = img._renderer && img._renderer.svg; // handle SVG Graphics
        elt = elt || (img.elt && img.elt.nodeName && (img.elt.nodeName.toLowerCase() === 'svg') && img.elt); // SVGElement
        elt = elt || (img.nodeName && (img.nodeName.toLowerCase() == 'svg') && img); // <svg>
        if (elt) {
            // it's <svg> element, let's handle it
            elt = elt.cloneNode(true);
            elt.setAttribute('width', w);
            elt.setAttribute('height', h);
            elt.setAttribute('x', x);
            elt.setAttribute('y', y);
            this.appendChild(elt);
        } else {
            p5.Renderer2D.prototype.image.apply(this, arguments);
        }
    };

    p5.RendererSVG = RendererSVG;
};

},{"svgcanvas":3}],10:[function(require,module,exports){
module.exports = function(p5) {
    var SVGFilters = function() {};

    var SVGElement = p5.SVGElement;

    var generateID = function() {
        return Date.now().toString() + Math.random().toString().replace(/0\./, '');
    };

    // @private
    // We have to build a filter for each element
    // the `filter: f1 f2` and svg param is not supported by many browsers
    // so we can just modify the filter def to do so
    SVGFilters.apply = function(svgElement, func, arg) {
        // get filters
        var filters = svgElement.attribute('data-p5-svg-filters') || '[]';
        filters = JSON.parse(filters);
        if (func) {
            filters.push([func, arg]);
        }
        svgElement.attribute('data-p5-svg-filters', JSON.stringify(filters));

        if (filters.length === 0) {
            svgElement.attribute('filter', null);
            return;
        }

        // generate filters chain
        filters = filters.map(function(filter, index) {
            var inGraphics = index === 0 ? 'SourceGraphic' : ('result-' + (index - 1));
            var resultGraphics = 'result-' + index;
            return SVGFilters[filter[0]].call(null, inGraphics, resultGraphics, filter[1]);
        });

        // get filter id for this element or create one
        var filterid = svgElement.attribute('data-p5-svg-filter-id');
        if (!filterid) {
            filterid = 'p5-svg-' + generateID();
            svgElement.attribute('data-p5-svg-filter-id', filterid);
        }
        // Note that when filters is [], we will remove filter attr
        // So, here, we write this attr every time
        svgElement.attribute('filter', 'url(#' + filterid + ')');

        // create <filter>
        var filter = SVGElement.create('filter', {id: filterid});
        filters.forEach(function(elt) {
            if (!Array.isArray(elt)) {
                elt = [elt];
            }
            elt.forEach(function(elt) {
                filter.append(elt);
            });
        });

        // get defs
        var defs = svgElement._getDefs();
        var oldfilter = defs.query('#' + filterid)[0];
        if (!oldfilter) {
            defs.append(filter);
        } else {
            oldfilter.elt.parentNode.replaceChild(filter.elt, oldfilter.elt);
        }
    };

    SVGFilters.blur = function(inGraphics, resultGraphics, val) {
        return SVGElement.create('feGaussianBlur', {
            stdDeviation: val,
            in: inGraphics,
            result: resultGraphics,
            'color-interpolation-filters': 'sRGB'
        });
    };

    // See also: http://www.w3.org/TR/SVG11/filters.html#feColorMatrixElement
    // See also: http://stackoverflow.com/questions/21977929/match-colors-in-fecolormatrix-filter
    SVGFilters.colorMatrix = function(inGraphics, resultGraphics, matrix) {
        return SVGElement.create('feColorMatrix', {
            type: 'matrix',
            values: matrix.join(' '),
            'color-interpolation-filters': 'sRGB',
            in: inGraphics,
            result: resultGraphics
        });
    };

    // Here we use CIE luminance for RGB
    SVGFilters.gray = function(inGraphics, resultGraphics) {
        var matrix = [
            0.2126, 0.7152, 0.0722, 0, 0, // R'
            0.2126, 0.7152, 0.0722, 0, 0, // G'
            0.2126, 0.7152, 0.0722, 0, 0, // B'
            0, 0, 0, 1, 0 // A'
        ];
        return SVGFilters.colorMatrix(inGraphics, resultGraphics, matrix);
    };

    SVGFilters.threshold = function(inGraphics, resultGraphics, val) {
        var elements = [];
        elements.push(SVGFilters.gray(inGraphics, resultGraphics + '-tmp'));
        var componentTransfer = SVGElement.create('feComponentTransfer', {
            'in': resultGraphics + '-tmp',
            result: resultGraphics
        });
        var thresh = Math.floor(val * 255);
        ['R', 'G', 'B'].forEach(function(channel) {
            // Note that original value is from 0 to 1
            var func = SVGElement.create('feFunc' + channel, {
                type: 'linear',
                slope: 255, // all non-zero * 255
                intercept: (thresh - 1) * -1
            });
            componentTransfer.append(func);
        });
        elements.push(componentTransfer);
        return elements;
    };

    SVGFilters.invert = function(inGraphics, resultGraphics) {
        var matrix = [
            -1, 0, 0, 0, 1,
            0, -1, 0, 0, 1,
            0, 0, -1, 0, 1,
            0, 0, 0, 1, 0
        ];
        return SVGFilters.colorMatrix(inGraphics, resultGraphics, matrix);
    };

    SVGFilters.opaque = function(inGraphics, resultGraphics) {
        var matrix = [
            1, 0, 0, 0, 0, // original R
            0, 1, 0, 0, 0, // original G
            0, 0, 1, 0, 0, // original B
            0, 0, 0, 0, 1 // set A to 1
        ];
        return SVGFilters.colorMatrix(inGraphics, resultGraphics, matrix);
    };

    /**
     * Generate discrete table values based on the given color map function
     *
     * @private
     * @param {Function} fn - Function to map channel values (val ∈ [0, 255])
     * @see http://www.w3.org/TR/SVG/filters.html#feComponentTransferElement
     */
    SVGFilters._discreteTableValues = function(fn) {
        var table = [];
        for (var val = 0; val < 256; val++) {
            var newval = fn(val);
            table.push(newval / 255); // map to ∈ [0, 1]
        }
        return table;
    };

    /**
     * Limits each channel of the image to the number of colors specified as
     * the parameter. The parameter can be set to values between 2 and 255, but
     * results are most noticeable in the lower ranges.
     *
     * Adapted from p5's Filters.posterize
     */
    SVGFilters.posterize = function(inGraphics, resultGraphics, level) {
        level = parseInt(level, 10);
        if ((level < 2) || (level > 255)) {
            throw new Error(
                'Level must be greater than 2 and less than 255 for posterize'
            );
        }

        var tableValues = SVGFilters._discreteTableValues(function(val) {
            return (((val * level) >> 8) * 255) / (level - 1);
        });

        var componentTransfer = SVGElement.create('feComponentTransfer', {
            'in': inGraphics,
            result: resultGraphics,
            'color-interpolation-filters': 'sRGB'
        });
        ['R', 'G', 'B'].forEach(function(channel) {
            var func = SVGElement.create('feFunc' + channel, {
                type: 'discrete',
                tableValues: tableValues.join(' ')
            });
            componentTransfer.append(func);
        });

        return componentTransfer;
    };

    SVGFilters._blendOffset = function(inGraphics, resultGraphics, mode) {
        var elements = [];
        [
            ['left', -1, 0],
            ['right', 1, 0],
            ['up', 0, -1],
            ['down', 0, 1]
        ].forEach(function(neighbor) {
            elements.push(SVGElement.create('feOffset', {
                'in': inGraphics,
                result: resultGraphics + '-' + neighbor[0],
                dx: neighbor[1],
                dy: neighbor[2]
            }));
        });
        [
            [null, inGraphics],
            [resultGraphics + '-left', resultGraphics + '-tmp-0'],
            [resultGraphics + '-right', resultGraphics + '-tmp-1'],
            [resultGraphics + '-up', resultGraphics + '-tmp-2'],
            [resultGraphics + '-down', resultGraphics + '-tmp-3']
        ].forEach(function(layer, i, layers) {
            if (i === 0) {
                return;
            }
            elements.push(SVGElement.create('feBlend', {
                'in': layers[i - 1][1],
                in2: layer[0],
                result: layer[1],
                mode: mode
            }));
        });
        return elements;
    };

    /**
     * Increases the bright areas in an image
     *
     * Will create 4 offset layer and blend them using darken mode
     */
    SVGFilters.erode = function(inGraphics, resultGraphics) {
        return SVGFilters._blendOffset(inGraphics, resultGraphics, 'darken');
    };

    SVGFilters.dilate = function(inGraphics, resultGraphics) {
        return SVGFilters._blendOffset(inGraphics, resultGraphics, 'lighten');
    };

    p5.SVGFilters = SVGFilters;

    return SVGFilters;
};

},{}],11:[function(require,module,exports){
var constants = require('./constants');
var SVGCanvas = require('svgcanvas');

module.exports = function(p5) {
    // patch p5.Graphics for SVG
    var _graphics = p5.Graphics;
    p5.Graphics = function(w, h, renderer, pInst) {
        var args = arguments;
        _graphics.apply(this, args);
        if (renderer === constants.SVG) {
            // replace <canvas> with <svg>
            var c = this._renderer.elt;
            this._renderer = new p5.RendererSVG(c, pInst, false); // replace renderer
            c = this._renderer.elt;
            this.elt = c; // replace this.elt
            // do default again
            this._renderer.resize(w, h);
            this._renderer._applyDefaults();
        }
        return this;
    };
    p5.Graphics.prototype = _graphics.prototype;

    /**
     * Due to a known issue (image is not surely ready before onload fires),
     * we have no way to draw SVG element synchronously.
     * So, this method will load a SVG Graphics
     * and then convert it to Canvas Graphics asynchronously
     *
     * @see https://github.com/zenozeng/p5.js-svg/issues/78
     *
     * @function loadGraphics
     * @memberof p5.prototype
     * @param {p5.Graphics} graphics the p5.Grphaics object
     * @param {Function(p5.Graphics)} [successCallback] Function to be called once
     *                                 the SVG Graphics is loaded. Will be passed the
     *                                 p5.Graphics.
     * @param {Function(Event)}    [failureCallback] called with event error.
     *
     * @example
     * pg = createGraphics(100, 100, SVG);
     * background(200);
     * pg.background(100);
     * pg.ellipse(pg.width/2, pg.height/2, 50, 50);
     * loadGraphics(pg, function(pgCanvas) {
     *      image(pgCanvas, 50, 50);
     *      image(pgCanvas, 0, 0, 50, 50);
     * });
     *
     */
    p5.prototype.loadGraphics = function(graphics, successCallback, failureCallback) {
        if (graphics._renderer.svg) {
            var svg = graphics._renderer.svg;
            var url = SVGCanvas.prototype.toDataURL.call(graphics._renderer.elt, 'image/svg+xml');
            var pg = this.createGraphics(graphics.width, graphics.height, constants.SVG);
            // also copy SVG, so we can keep vector SVG when image(pg) in SVG runtime
            pg._renderer.svg = svg.cloneNode(true);
            pg.loadImage(url, function(img) {
                pg.image(img);
                successCallback(pg);
            }, failureCallback);
        } else {
            successCallback(graphics);
        }
    };

    /**
     * Patched version of createCanvas
     *
     * use createCanvas(100, 100, SVG) to create SVG canvas.
     *
     * Creates a SVG element in the document, and sets its width and
     * height in pixels. This method should be called only once at
     * the start of setup.
     * @function createCanvas
     * @memberof p5.prototype
     * @param {Number} width - Width (in px) for SVG Element
     * @param {Number} height - Height (in px) for SVG Element
     * @return {Graphics}
     */
    var _createCanvas = p5.prototype.createCanvas;
    p5.prototype.createCanvas = function(w, h, renderer) {
        var graphics = _createCanvas.apply(this, arguments);
        if (renderer === constants.SVG) {
            var c = graphics.elt;
            this._setProperty('_renderer', new p5.RendererSVG(c, this, true));
            this._isdefaultGraphics = true;
            this._renderer.resize(w, h);
            this._renderer._applyDefaults();
        }
        return this._renderer;
    };
};

},{"./constants":4,"svgcanvas":3}],12:[function(require,module,exports){
require('../src/index.js')(p5);

},{"../src/index.js":7}]},{},[12]);
});