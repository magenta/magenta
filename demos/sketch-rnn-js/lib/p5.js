/*! p5.js v0.5.2 June 17, 2016 */
(function(f){if(typeof exports==="object"&&typeof module!=="undefined"){module.exports=f()}else if(typeof define==="function"&&define.amd){define([],f)}else{var g;if(typeof window!=="undefined"){g=window}else if(typeof global!=="undefined"){g=global}else if(typeof self!=="undefined"){g=self}else{g=this}g.p5 = f()}})(function(){var define,module,exports;return (function e(t,n,r){function s(o,u){if(!n[o]){if(!t[o]){var a=typeof require=="function"&&require;if(!u&&a)return a(o,!0);if(i)return i(o,!0);var f=new Error("Cannot find module '"+o+"'");throw f.code="MODULE_NOT_FOUND",f}var l=n[o]={exports:{}};t[o][0].call(l.exports,function(e){var n=t[o][1][e];return s(n?n:e)},l,l.exports,e,t,n,r)}return n[o].exports}var i=typeof require=="function"&&require;for(var o=0;o<r.length;o++)s(r[o]);return s})({1:[function(_dereq_,module,exports){

},{}],2:[function(_dereq_,module,exports){
// Run-time checking of preconditions.

'use strict';

// Precondition function that checks if the given predicate is true.
// If not, it will throw an error.
exports.argument = function(predicate, message) {
    if (!predicate) {
        throw new Error(message);
    }
};

// Precondition function that checks if the given assertion is true.
// If not, it will throw an error.
exports.assert = exports.argument;

},{}],3:[function(_dereq_,module,exports){
// Drawing utility functions.

'use strict';

// Draw a line on the given context from point `x1,y1` to point `x2,y2`.
function line(ctx, x1, y1, x2, y2) {
    ctx.beginPath();
    ctx.moveTo(x1, y1);
    ctx.lineTo(x2, y2);
    ctx.stroke();
}

exports.line = line;

},{}],4:[function(_dereq_,module,exports){
// Glyph encoding

'use strict';

var cffStandardStrings = [
    '.notdef', 'space', 'exclam', 'quotedbl', 'numbersign', 'dollar', 'percent', 'ampersand', 'quoteright',
    'parenleft', 'parenright', 'asterisk', 'plus', 'comma', 'hyphen', 'period', 'slash', 'zero', 'one', 'two',
    'three', 'four', 'five', 'six', 'seven', 'eight', 'nine', 'colon', 'semicolon', 'less', 'equal', 'greater',
    'question', 'at', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S',
    'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 'bracketleft', 'backslash', 'bracketright', 'asciicircum', 'underscore',
    'quoteleft', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't',
    'u', 'v', 'w', 'x', 'y', 'z', 'braceleft', 'bar', 'braceright', 'asciitilde', 'exclamdown', 'cent', 'sterling',
    'fraction', 'yen', 'florin', 'section', 'currency', 'quotesingle', 'quotedblleft', 'guillemotleft',
    'guilsinglleft', 'guilsinglright', 'fi', 'fl', 'endash', 'dagger', 'daggerdbl', 'periodcentered', 'paragraph',
    'bullet', 'quotesinglbase', 'quotedblbase', 'quotedblright', 'guillemotright', 'ellipsis', 'perthousand',
    'questiondown', 'grave', 'acute', 'circumflex', 'tilde', 'macron', 'breve', 'dotaccent', 'dieresis', 'ring',
    'cedilla', 'hungarumlaut', 'ogonek', 'caron', 'emdash', 'AE', 'ordfeminine', 'Lslash', 'Oslash', 'OE',
    'ordmasculine', 'ae', 'dotlessi', 'lslash', 'oslash', 'oe', 'germandbls', 'onesuperior', 'logicalnot', 'mu',
    'trademark', 'Eth', 'onehalf', 'plusminus', 'Thorn', 'onequarter', 'divide', 'brokenbar', 'degree', 'thorn',
    'threequarters', 'twosuperior', 'registered', 'minus', 'eth', 'multiply', 'threesuperior', 'copyright',
    'Aacute', 'Acircumflex', 'Adieresis', 'Agrave', 'Aring', 'Atilde', 'Ccedilla', 'Eacute', 'Ecircumflex',
    'Edieresis', 'Egrave', 'Iacute', 'Icircumflex', 'Idieresis', 'Igrave', 'Ntilde', 'Oacute', 'Ocircumflex',
    'Odieresis', 'Ograve', 'Otilde', 'Scaron', 'Uacute', 'Ucircumflex', 'Udieresis', 'Ugrave', 'Yacute',
    'Ydieresis', 'Zcaron', 'aacute', 'acircumflex', 'adieresis', 'agrave', 'aring', 'atilde', 'ccedilla', 'eacute',
    'ecircumflex', 'edieresis', 'egrave', 'iacute', 'icircumflex', 'idieresis', 'igrave', 'ntilde', 'oacute',
    'ocircumflex', 'odieresis', 'ograve', 'otilde', 'scaron', 'uacute', 'ucircumflex', 'udieresis', 'ugrave',
    'yacute', 'ydieresis', 'zcaron', 'exclamsmall', 'Hungarumlautsmall', 'dollaroldstyle', 'dollarsuperior',
    'ampersandsmall', 'Acutesmall', 'parenleftsuperior', 'parenrightsuperior', '266 ff', 'onedotenleader',
    'zerooldstyle', 'oneoldstyle', 'twooldstyle', 'threeoldstyle', 'fouroldstyle', 'fiveoldstyle', 'sixoldstyle',
    'sevenoldstyle', 'eightoldstyle', 'nineoldstyle', 'commasuperior', 'threequartersemdash', 'periodsuperior',
    'questionsmall', 'asuperior', 'bsuperior', 'centsuperior', 'dsuperior', 'esuperior', 'isuperior', 'lsuperior',
    'msuperior', 'nsuperior', 'osuperior', 'rsuperior', 'ssuperior', 'tsuperior', 'ff', 'ffi', 'ffl',
    'parenleftinferior', 'parenrightinferior', 'Circumflexsmall', 'hyphensuperior', 'Gravesmall', 'Asmall',
    'Bsmall', 'Csmall', 'Dsmall', 'Esmall', 'Fsmall', 'Gsmall', 'Hsmall', 'Ismall', 'Jsmall', 'Ksmall', 'Lsmall',
    'Msmall', 'Nsmall', 'Osmall', 'Psmall', 'Qsmall', 'Rsmall', 'Ssmall', 'Tsmall', 'Usmall', 'Vsmall', 'Wsmall',
    'Xsmall', 'Ysmall', 'Zsmall', 'colonmonetary', 'onefitted', 'rupiah', 'Tildesmall', 'exclamdownsmall',
    'centoldstyle', 'Lslashsmall', 'Scaronsmall', 'Zcaronsmall', 'Dieresissmall', 'Brevesmall', 'Caronsmall',
    'Dotaccentsmall', 'Macronsmall', 'figuredash', 'hypheninferior', 'Ogoneksmall', 'Ringsmall', 'Cedillasmall',
    'questiondownsmall', 'oneeighth', 'threeeighths', 'fiveeighths', 'seveneighths', 'onethird', 'twothirds',
    'zerosuperior', 'foursuperior', 'fivesuperior', 'sixsuperior', 'sevensuperior', 'eightsuperior', 'ninesuperior',
    'zeroinferior', 'oneinferior', 'twoinferior', 'threeinferior', 'fourinferior', 'fiveinferior', 'sixinferior',
    'seveninferior', 'eightinferior', 'nineinferior', 'centinferior', 'dollarinferior', 'periodinferior',
    'commainferior', 'Agravesmall', 'Aacutesmall', 'Acircumflexsmall', 'Atildesmall', 'Adieresissmall',
    'Aringsmall', 'AEsmall', 'Ccedillasmall', 'Egravesmall', 'Eacutesmall', 'Ecircumflexsmall', 'Edieresissmall',
    'Igravesmall', 'Iacutesmall', 'Icircumflexsmall', 'Idieresissmall', 'Ethsmall', 'Ntildesmall', 'Ogravesmall',
    'Oacutesmall', 'Ocircumflexsmall', 'Otildesmall', 'Odieresissmall', 'OEsmall', 'Oslashsmall', 'Ugravesmall',
    'Uacutesmall', 'Ucircumflexsmall', 'Udieresissmall', 'Yacutesmall', 'Thornsmall', 'Ydieresissmall', '001.000',
    '001.001', '001.002', '001.003', 'Black', 'Bold', 'Book', 'Light', 'Medium', 'Regular', 'Roman', 'Semibold'];

var cffStandardEncoding = [
    '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '',
    '', '', '', '', 'space', 'exclam', 'quotedbl', 'numbersign', 'dollar', 'percent', 'ampersand', 'quoteright',
    'parenleft', 'parenright', 'asterisk', 'plus', 'comma', 'hyphen', 'period', 'slash', 'zero', 'one', 'two',
    'three', 'four', 'five', 'six', 'seven', 'eight', 'nine', 'colon', 'semicolon', 'less', 'equal', 'greater',
    'question', 'at', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S',
    'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 'bracketleft', 'backslash', 'bracketright', 'asciicircum', 'underscore',
    'quoteleft', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't',
    'u', 'v', 'w', 'x', 'y', 'z', 'braceleft', 'bar', 'braceright', 'asciitilde', '', '', '', '', '', '', '', '',
    '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '',
    'exclamdown', 'cent', 'sterling', 'fraction', 'yen', 'florin', 'section', 'currency', 'quotesingle',
    'quotedblleft', 'guillemotleft', 'guilsinglleft', 'guilsinglright', 'fi', 'fl', '', 'endash', 'dagger',
    'daggerdbl', 'periodcentered', '', 'paragraph', 'bullet', 'quotesinglbase', 'quotedblbase', 'quotedblright',
    'guillemotright', 'ellipsis', 'perthousand', '', 'questiondown', '', 'grave', 'acute', 'circumflex', 'tilde',
    'macron', 'breve', 'dotaccent', 'dieresis', '', 'ring', 'cedilla', '', 'hungarumlaut', 'ogonek', 'caron',
    'emdash', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', 'AE', '', 'ordfeminine', '', '', '',
    '', 'Lslash', 'Oslash', 'OE', 'ordmasculine', '', '', '', '', '', 'ae', '', '', '', 'dotlessi', '', '',
    'lslash', 'oslash', 'oe', 'germandbls'];

var cffExpertEncoding = [
    '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '',
    '', '', '', '', 'space', 'exclamsmall', 'Hungarumlautsmall', '', 'dollaroldstyle', 'dollarsuperior',
    'ampersandsmall', 'Acutesmall', 'parenleftsuperior', 'parenrightsuperior', 'twodotenleader', 'onedotenleader',
    'comma', 'hyphen', 'period', 'fraction', 'zerooldstyle', 'oneoldstyle', 'twooldstyle', 'threeoldstyle',
    'fouroldstyle', 'fiveoldstyle', 'sixoldstyle', 'sevenoldstyle', 'eightoldstyle', 'nineoldstyle', 'colon',
    'semicolon', 'commasuperior', 'threequartersemdash', 'periodsuperior', 'questionsmall', '', 'asuperior',
    'bsuperior', 'centsuperior', 'dsuperior', 'esuperior', '', '', 'isuperior', '', '', 'lsuperior', 'msuperior',
    'nsuperior', 'osuperior', '', '', 'rsuperior', 'ssuperior', 'tsuperior', '', 'ff', 'fi', 'fl', 'ffi', 'ffl',
    'parenleftinferior', '', 'parenrightinferior', 'Circumflexsmall', 'hyphensuperior', 'Gravesmall', 'Asmall',
    'Bsmall', 'Csmall', 'Dsmall', 'Esmall', 'Fsmall', 'Gsmall', 'Hsmall', 'Ismall', 'Jsmall', 'Ksmall', 'Lsmall',
    'Msmall', 'Nsmall', 'Osmall', 'Psmall', 'Qsmall', 'Rsmall', 'Ssmall', 'Tsmall', 'Usmall', 'Vsmall', 'Wsmall',
    'Xsmall', 'Ysmall', 'Zsmall', 'colonmonetary', 'onefitted', 'rupiah', 'Tildesmall', '', '', '', '', '', '', '',
    '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '',
    'exclamdownsmall', 'centoldstyle', 'Lslashsmall', '', '', 'Scaronsmall', 'Zcaronsmall', 'Dieresissmall',
    'Brevesmall', 'Caronsmall', '', 'Dotaccentsmall', '', '', 'Macronsmall', '', '', 'figuredash', 'hypheninferior',
    '', '', 'Ogoneksmall', 'Ringsmall', 'Cedillasmall', '', '', '', 'onequarter', 'onehalf', 'threequarters',
    'questiondownsmall', 'oneeighth', 'threeeighths', 'fiveeighths', 'seveneighths', 'onethird', 'twothirds', '',
    '', 'zerosuperior', 'onesuperior', 'twosuperior', 'threesuperior', 'foursuperior', 'fivesuperior',
    'sixsuperior', 'sevensuperior', 'eightsuperior', 'ninesuperior', 'zeroinferior', 'oneinferior', 'twoinferior',
    'threeinferior', 'fourinferior', 'fiveinferior', 'sixinferior', 'seveninferior', 'eightinferior',
    'nineinferior', 'centinferior', 'dollarinferior', 'periodinferior', 'commainferior', 'Agravesmall',
    'Aacutesmall', 'Acircumflexsmall', 'Atildesmall', 'Adieresissmall', 'Aringsmall', 'AEsmall', 'Ccedillasmall',
    'Egravesmall', 'Eacutesmall', 'Ecircumflexsmall', 'Edieresissmall', 'Igravesmall', 'Iacutesmall',
    'Icircumflexsmall', 'Idieresissmall', 'Ethsmall', 'Ntildesmall', 'Ogravesmall', 'Oacutesmall',
    'Ocircumflexsmall', 'Otildesmall', 'Odieresissmall', 'OEsmall', 'Oslashsmall', 'Ugravesmall', 'Uacutesmall',
    'Ucircumflexsmall', 'Udieresissmall', 'Yacutesmall', 'Thornsmall', 'Ydieresissmall'];

var standardNames = [
    '.notdef', '.null', 'nonmarkingreturn', 'space', 'exclam', 'quotedbl', 'numbersign', 'dollar', 'percent',
    'ampersand', 'quotesingle', 'parenleft', 'parenright', 'asterisk', 'plus', 'comma', 'hyphen', 'period', 'slash',
    'zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine', 'colon', 'semicolon', 'less',
    'equal', 'greater', 'question', 'at', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O',
    'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 'bracketleft', 'backslash', 'bracketright',
    'asciicircum', 'underscore', 'grave', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o',
    'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z', 'braceleft', 'bar', 'braceright', 'asciitilde',
    'Adieresis', 'Aring', 'Ccedilla', 'Eacute', 'Ntilde', 'Odieresis', 'Udieresis', 'aacute', 'agrave',
    'acircumflex', 'adieresis', 'atilde', 'aring', 'ccedilla', 'eacute', 'egrave', 'ecircumflex', 'edieresis',
    'iacute', 'igrave', 'icircumflex', 'idieresis', 'ntilde', 'oacute', 'ograve', 'ocircumflex', 'odieresis',
    'otilde', 'uacute', 'ugrave', 'ucircumflex', 'udieresis', 'dagger', 'degree', 'cent', 'sterling', 'section',
    'bullet', 'paragraph', 'germandbls', 'registered', 'copyright', 'trademark', 'acute', 'dieresis', 'notequal',
    'AE', 'Oslash', 'infinity', 'plusminus', 'lessequal', 'greaterequal', 'yen', 'mu', 'partialdiff', 'summation',
    'product', 'pi', 'integral', 'ordfeminine', 'ordmasculine', 'Omega', 'ae', 'oslash', 'questiondown',
    'exclamdown', 'logicalnot', 'radical', 'florin', 'approxequal', 'Delta', 'guillemotleft', 'guillemotright',
    'ellipsis', 'nonbreakingspace', 'Agrave', 'Atilde', 'Otilde', 'OE', 'oe', 'endash', 'emdash', 'quotedblleft',
    'quotedblright', 'quoteleft', 'quoteright', 'divide', 'lozenge', 'ydieresis', 'Ydieresis', 'fraction',
    'currency', 'guilsinglleft', 'guilsinglright', 'fi', 'fl', 'daggerdbl', 'periodcentered', 'quotesinglbase',
    'quotedblbase', 'perthousand', 'Acircumflex', 'Ecircumflex', 'Aacute', 'Edieresis', 'Egrave', 'Iacute',
    'Icircumflex', 'Idieresis', 'Igrave', 'Oacute', 'Ocircumflex', 'apple', 'Ograve', 'Uacute', 'Ucircumflex',
    'Ugrave', 'dotlessi', 'circumflex', 'tilde', 'macron', 'breve', 'dotaccent', 'ring', 'cedilla', 'hungarumlaut',
    'ogonek', 'caron', 'Lslash', 'lslash', 'Scaron', 'scaron', 'Zcaron', 'zcaron', 'brokenbar', 'Eth', 'eth',
    'Yacute', 'yacute', 'Thorn', 'thorn', 'minus', 'multiply', 'onesuperior', 'twosuperior', 'threesuperior',
    'onehalf', 'onequarter', 'threequarters', 'franc', 'Gbreve', 'gbreve', 'Idotaccent', 'Scedilla', 'scedilla',
    'Cacute', 'cacute', 'Ccaron', 'ccaron', 'dcroat'];

// This is the encoding used for fonts created from scratch.
// It loops through all glyphs and finds the appropriate unicode value.
// Since it's linear time, other encodings will be faster.
function DefaultEncoding(font) {
    this.font = font;
}

DefaultEncoding.prototype.charToGlyphIndex = function(c) {
    var code = c.charCodeAt(0);
    var glyphs = this.font.glyphs;
    if (glyphs) {
        for (var i = 0; i < glyphs.length; i += 1) {
            var glyph = glyphs.get(i);
            for (var j = 0; j < glyph.unicodes.length; j += 1) {
                if (glyph.unicodes[j] === code) {
                    return i;
                }
            }
        }
    } else {
        return null;
    }
};

function CmapEncoding(cmap) {
    this.cmap = cmap;
}

CmapEncoding.prototype.charToGlyphIndex = function(c) {
    return this.cmap.glyphIndexMap[c.charCodeAt(0)] || 0;
};

function CffEncoding(encoding, charset) {
    this.encoding = encoding;
    this.charset = charset;
}

CffEncoding.prototype.charToGlyphIndex = function(s) {
    var code = s.charCodeAt(0);
    var charName = this.encoding[code];
    return this.charset.indexOf(charName);
};

function GlyphNames(post) {
    var i;
    switch (post.version) {
    case 1:
        this.names = exports.standardNames.slice();
        break;
    case 2:
        this.names = new Array(post.numberOfGlyphs);
        for (i = 0; i < post.numberOfGlyphs; i++) {
            if (post.glyphNameIndex[i] < exports.standardNames.length) {
                this.names[i] = exports.standardNames[post.glyphNameIndex[i]];
            } else {
                this.names[i] = post.names[post.glyphNameIndex[i] - exports.standardNames.length];
            }
        }

        break;
    case 2.5:
        this.names = new Array(post.numberOfGlyphs);
        for (i = 0; i < post.numberOfGlyphs; i++) {
            this.names[i] = exports.standardNames[i + post.glyphNameIndex[i]];
        }

        break;
    case 3:
        this.names = [];
        break;
    }
}

GlyphNames.prototype.nameToGlyphIndex = function(name) {
    return this.names.indexOf(name);
};

GlyphNames.prototype.glyphIndexToName = function(gid) {
    return this.names[gid];
};

function addGlyphNames(font) {
    var glyph;
    var glyphIndexMap = font.tables.cmap.glyphIndexMap;
    var charCodes = Object.keys(glyphIndexMap);

    for (var i = 0; i < charCodes.length; i += 1) {
        var c = charCodes[i];
        var glyphIndex = glyphIndexMap[c];
        glyph = font.glyphs.get(glyphIndex);
        glyph.addUnicode(parseInt(c));
    }

    for (i = 0; i < font.glyphs.length; i += 1) {
        glyph = font.glyphs.get(i);
        if (font.cffEncoding) {
            glyph.name = font.cffEncoding.charset[i];
        } else {
            glyph.name = font.glyphNames.glyphIndexToName(i);
        }
    }
}

exports.cffStandardStrings = cffStandardStrings;
exports.cffStandardEncoding = cffStandardEncoding;
exports.cffExpertEncoding = cffExpertEncoding;
exports.standardNames = standardNames;
exports.DefaultEncoding = DefaultEncoding;
exports.CmapEncoding = CmapEncoding;
exports.CffEncoding = CffEncoding;
exports.GlyphNames = GlyphNames;
exports.addGlyphNames = addGlyphNames;

},{}],5:[function(_dereq_,module,exports){
// The Font object

'use strict';

var path = _dereq_('./path');
var sfnt = _dereq_('./tables/sfnt');
var encoding = _dereq_('./encoding');
var glyphset = _dereq_('./glyphset');

// A Font represents a loaded OpenType font file.
// It contains a set of glyphs and methods to draw text on a drawing context,
// or to get a path representing the text.
function Font(options) {
    options = options || {};

    // OS X will complain if the names are empty, so we put a single space everywhere by default.
    this.familyName = options.familyName || ' ';
    this.styleName = options.styleName || ' ';
    this.designer = options.designer || ' ';
    this.designerURL = options.designerURL || ' ';
    this.manufacturer = options.manufacturer || ' ';
    this.manufacturerURL = options.manufacturerURL || ' ';
    this.license = options.license || ' ';
    this.licenseURL = options.licenseURL || ' ';
    this.version = options.version || 'Version 0.1';
    this.description = options.description || ' ';
    this.copyright = options.copyright || ' ';
    this.trademark = options.trademark || ' ';
    this.unitsPerEm = options.unitsPerEm || 1000;
    this.ascender = options.ascender;
    this.descender = options.descender;
    this.supported = true;
    this.glyphs = new glyphset.GlyphSet(this, options.glyphs || []);
    this.encoding = new encoding.DefaultEncoding(this);
    this.tables = {};
}

// Check if the font has a glyph for the given character.
Font.prototype.hasChar = function(c) {
    return this.encoding.charToGlyphIndex(c) !== null;
};

// Convert the given character to a single glyph index.
// Note that this function assumes that there is a one-to-one mapping between
// the given character and a glyph; for complex scripts this might not be the case.
Font.prototype.charToGlyphIndex = function(s) {
    return this.encoding.charToGlyphIndex(s);
};

// Convert the given character to a single Glyph object.
// Note that this function assumes that there is a one-to-one mapping between
// the given character and a glyph; for complex scripts this might not be the case.
Font.prototype.charToGlyph = function(c) {
    var glyphIndex = this.charToGlyphIndex(c);
    var glyph = this.glyphs.get(glyphIndex);
    if (!glyph) {
        // .notdef
        glyph = this.glyphs.get(0);
    }

    return glyph;
};

// Convert the given text to a list of Glyph objects.
// Note that there is no strict one-to-one mapping between characters and
// glyphs, so the list of returned glyphs can be larger or smaller than the
// length of the given string.
Font.prototype.stringToGlyphs = function(s) {
    var glyphs = [];
    for (var i = 0; i < s.length; i += 1) {
        var c = s[i];
        glyphs.push(this.charToGlyph(c));
    }

    return glyphs;
};

Font.prototype.nameToGlyphIndex = function(name) {
    return this.glyphNames.nameToGlyphIndex(name);
};

Font.prototype.nameToGlyph = function(name) {
    var glyphIndex = this.nametoGlyphIndex(name);
    var glyph = this.glyphs.get(glyphIndex);
    if (!glyph) {
        // .notdef
        glyph = this.glyphs.get(0);
    }

    return glyph;
};

Font.prototype.glyphIndexToName = function(gid) {
    if (!this.glyphNames.glyphIndexToName) {
        return '';
    }

    return this.glyphNames.glyphIndexToName(gid);
};

// Retrieve the value of the kerning pair between the left glyph (or its index)
// and the right glyph (or its index). If no kerning pair is found, return 0.
// The kerning value gets added to the advance width when calculating the spacing
// between glyphs.
Font.prototype.getKerningValue = function(leftGlyph, rightGlyph) {
    leftGlyph = leftGlyph.index || leftGlyph;
    rightGlyph = rightGlyph.index || rightGlyph;
    var gposKerning = this.getGposKerningValue;
    return gposKerning ? gposKerning(leftGlyph, rightGlyph) :
        (this.kerningPairs[leftGlyph + ',' + rightGlyph] || 0);
};

// Helper function that invokes the given callback for each glyph in the given text.
// The callback gets `(glyph, x, y, fontSize, options)`.
Font.prototype.forEachGlyph = function(text, x, y, fontSize, options, callback) {
    if (!this.supported) {
        return;
    }

    x = x !== undefined ? x : 0;
    y = y !== undefined ? y : 0;
    fontSize = fontSize !== undefined ? fontSize : 72;
    options = options || {};
    var kerning = options.kerning === undefined ? true : options.kerning;
    var fontScale = 1 / this.unitsPerEm * fontSize;
    var glyphs = this.stringToGlyphs(text);
    for (var i = 0; i < glyphs.length; i += 1) {
        var glyph = glyphs[i];
        callback(glyph, x, y, fontSize, options);
        if (glyph.advanceWidth) {
            x += glyph.advanceWidth * fontScale;
        }

        if (kerning && i < glyphs.length - 1) {
            var kerningValue = this.getKerningValue(glyph, glyphs[i + 1]);
            x += kerningValue * fontScale;
        }
    }
};

// Create a Path object that represents the given text.
//
// text - The text to create.
// x - Horizontal position of the beginning of the text. (default: 0)
// y - Vertical position of the *baseline* of the text. (default: 0)
// fontSize - Font size in pixels. We scale the glyph units by `1 / unitsPerEm * fontSize`. (default: 72)
// Options is an optional object that contains:
// - kerning - Whether to take kerning information into account. (default: true)
//
// Returns a Path object.
Font.prototype.getPath = function(text, x, y, fontSize, options) {
    var fullPath = new path.Path();
    this.forEachGlyph(text, x, y, fontSize, options, function(glyph, gX, gY, gFontSize) {
        var glyphPath = glyph.getPath(gX, gY, gFontSize);
        fullPath.extend(glyphPath);
    });

    return fullPath;
};

// Draw the text on the given drawing context.
//
// ctx - A 2D drawing context, like Canvas.
// text - The text to create.
// x - Horizontal position of the beginning of the text. (default: 0)
// y - Vertical position of the *baseline* of the text. (default: 0)
// fontSize - Font size in pixels. We scale the glyph units by `1 / unitsPerEm * fontSize`. (default: 72)
// Options is an optional object that contains:
// - kerning - Whether to take kerning information into account. (default: true)
Font.prototype.draw = function(ctx, text, x, y, fontSize, options) {
    this.getPath(text, x, y, fontSize, options).draw(ctx);
};

// Draw the points of all glyphs in the text.
// On-curve points will be drawn in blue, off-curve points will be drawn in red.
//
// ctx - A 2D drawing context, like Canvas.
// text - The text to create.
// x - Horizontal position of the beginning of the text. (default: 0)
// y - Vertical position of the *baseline* of the text. (default: 0)
// fontSize - Font size in pixels. We scale the glyph units by `1 / unitsPerEm * fontSize`. (default: 72)
// Options is an optional object that contains:
// - kerning - Whether to take kerning information into account. (default: true)
Font.prototype.drawPoints = function(ctx, text, x, y, fontSize, options) {
    this.forEachGlyph(text, x, y, fontSize, options, function(glyph, gX, gY, gFontSize) {
        glyph.drawPoints(ctx, gX, gY, gFontSize);
    });
};

// Draw lines indicating important font measurements for all glyphs in the text.
// Black lines indicate the origin of the coordinate system (point 0,0).
// Blue lines indicate the glyph bounding box.
// Green line indicates the advance width of the glyph.
//
// ctx - A 2D drawing context, like Canvas.
// text - The text to create.
// x - Horizontal position of the beginning of the text. (default: 0)
// y - Vertical position of the *baseline* of the text. (default: 0)
// fontSize - Font size in pixels. We scale the glyph units by `1 / unitsPerEm * fontSize`. (default: 72)
// Options is an optional object that contains:
// - kerning - Whether to take kerning information into account. (default: true)
Font.prototype.drawMetrics = function(ctx, text, x, y, fontSize, options) {
    this.forEachGlyph(text, x, y, fontSize, options, function(glyph, gX, gY, gFontSize) {
        glyph.drawMetrics(ctx, gX, gY, gFontSize);
    });
};

// Validate
Font.prototype.validate = function() {
    var warnings = [];
    var _this = this;

    function assert(predicate, message) {
        if (!predicate) {
            warnings.push(message);
        }
    }

    function assertStringAttribute(attrName) {
        assert(_this[attrName] && _this[attrName].trim().length > 0, 'No ' + attrName + ' specified.');
    }

    // Identification information
    assertStringAttribute('familyName');
    assertStringAttribute('weightName');
    assertStringAttribute('manufacturer');
    assertStringAttribute('copyright');
    assertStringAttribute('version');

    // Dimension information
    assert(this.unitsPerEm > 0, 'No unitsPerEm specified.');
};

// Convert the font object to a SFNT data structure.
// This structure contains all the necessary tables and metadata to create a binary OTF file.
Font.prototype.toTables = function() {
    return sfnt.fontToTable(this);
};

Font.prototype.toBuffer = function() {
    var sfntTable = this.toTables();
    var bytes = sfntTable.encode();
    var buffer = new ArrayBuffer(bytes.length);
    var intArray = new Uint8Array(buffer);
    for (var i = 0; i < bytes.length; i++) {
        intArray[i] = bytes[i];
    }

    return buffer;
};

// Initiate a download of the OpenType font.
Font.prototype.download = function() {
    var fileName = this.familyName.replace(/\s/g, '') + '-' + this.styleName + '.otf';
    var buffer = this.toBuffer();

    window.requestFileSystem = window.requestFileSystem || window.webkitRequestFileSystem;
    window.requestFileSystem(window.TEMPORARY, buffer.byteLength, function(fs) {
        fs.root.getFile(fileName, {create: true}, function(fileEntry) {
            fileEntry.createWriter(function(writer) {
                var dataView = new DataView(buffer);
                var blob = new Blob([dataView], {type: 'font/opentype'});
                writer.write(blob);

                writer.addEventListener('writeend', function() {
                    // Navigating to the file will download it.
                    location.href = fileEntry.toURL();
                }, false);
            });
        });
    },

    function(err) {
        throw err;
    });
};

exports.Font = Font;

},{"./encoding":4,"./glyphset":7,"./path":10,"./tables/sfnt":25}],6:[function(_dereq_,module,exports){
// The Glyph object

'use strict';

var check = _dereq_('./check');
var draw = _dereq_('./draw');
var path = _dereq_('./path');

function getPathDefinition(glyph, path) {
    var _path = path || { commands: [] };
    return {
        configurable: true,

        get: function() {
            if (typeof _path === 'function') {
                _path = _path();
            }

            return _path;
        },

        set: function(p) {
            _path = p;
        }
    };
}

// A Glyph is an individual mark that often corresponds to a character.
// Some glyphs, such as ligatures, are a combination of many characters.
// Glyphs are the basic building blocks of a font.
//
// The `Glyph` class contains utility methods for drawing the path and its points.
function Glyph(options) {
    // By putting all the code on a prototype function (which is only declared once)
    // we reduce the memory requirements for larger fonts by some 2%
    this.bindConstructorValues(options);
}

Glyph.prototype.bindConstructorValues = function(options) {
    this.index = options.index || 0;

    // These three values cannnot be deferred for memory optimization:
    this.name = options.name || null;
    this.unicode = options.unicode || undefined;
    this.unicodes = options.unicodes || options.unicode !== undefined ? [options.unicode] : [];

    // But by binding these values only when necessary, we reduce can
    // the memory requirements by almost 3% for larger fonts.
    if (options.xMin) {
        this.xMin = options.xMin;
    }

    if (options.yMin) {
        this.yMin = options.yMin;
    }

    if (options.xMax) {
        this.xMax = options.xMax;
    }

    if (options.yMax) {
        this.yMax = options.yMax;
    }

    if (options.advanceWidth) {
        this.advanceWidth = options.advanceWidth;
    }

    // The path for a glyph is the most memory intensive, and is bound as a value
    // with a getter/setter to ensure we actually do path parsing only once the
    // path is actually needed by anything.
    Object.defineProperty(this, 'path', getPathDefinition(this, options.path));
};

Glyph.prototype.addUnicode = function(unicode) {
    if (this.unicodes.length === 0) {
        this.unicode = unicode;
    }

    this.unicodes.push(unicode);
};

// Convert the glyph to a Path we can draw on a drawing context.
//
// x - Horizontal position of the glyph. (default: 0)
// y - Vertical position of the *baseline* of the glyph. (default: 0)
// fontSize - Font size, in pixels (default: 72).
Glyph.prototype.getPath = function(x, y, fontSize) {
    x = x !== undefined ? x : 0;
    y = y !== undefined ? y : 0;
    fontSize = fontSize !== undefined ? fontSize : 72;
    var scale = 1 / this.path.unitsPerEm * fontSize;
    var p = new path.Path();
    var commands = this.path.commands;
    for (var i = 0; i < commands.length; i += 1) {
        var cmd = commands[i];
        if (cmd.type === 'M') {
            p.moveTo(x + (cmd.x * scale), y + (-cmd.y * scale));
        } else if (cmd.type === 'L') {
            p.lineTo(x + (cmd.x * scale), y + (-cmd.y * scale));
        } else if (cmd.type === 'Q') {
            p.quadraticCurveTo(x + (cmd.x1 * scale), y + (-cmd.y1 * scale),
                               x + (cmd.x * scale), y + (-cmd.y * scale));
        } else if (cmd.type === 'C') {
            p.curveTo(x + (cmd.x1 * scale), y + (-cmd.y1 * scale),
                      x + (cmd.x2 * scale), y + (-cmd.y2 * scale),
                      x + (cmd.x * scale), y + (-cmd.y * scale));
        } else if (cmd.type === 'Z') {
            p.closePath();
        }
    }

    return p;
};

// Split the glyph into contours.
// This function is here for backwards compatibility, and to
// provide raw access to the TrueType glyph outlines.
Glyph.prototype.getContours = function() {
    if (this.points === undefined) {
        return [];
    }

    var contours = [];
    var currentContour = [];
    for (var i = 0; i < this.points.length; i += 1) {
        var pt = this.points[i];
        currentContour.push(pt);
        if (pt.lastPointOfContour) {
            contours.push(currentContour);
            currentContour = [];
        }
    }

    check.argument(currentContour.length === 0, 'There are still points left in the current contour.');
    return contours;
};

// Calculate the xMin/yMin/xMax/yMax/lsb/rsb for a Glyph.
Glyph.prototype.getMetrics = function() {
    var commands = this.path.commands;
    var xCoords = [];
    var yCoords = [];
    for (var i = 0; i < commands.length; i += 1) {
        var cmd = commands[i];
        if (cmd.type !== 'Z') {
            xCoords.push(cmd.x);
            yCoords.push(cmd.y);
        }

        if (cmd.type === 'Q' || cmd.type === 'C') {
            xCoords.push(cmd.x1);
            yCoords.push(cmd.y1);
        }

        if (cmd.type === 'C') {
            xCoords.push(cmd.x2);
            yCoords.push(cmd.y2);
        }
    }

    var metrics = {
        xMin: Math.min.apply(null, xCoords),
        yMin: Math.min.apply(null, yCoords),
        xMax: Math.max.apply(null, xCoords),
        yMax: Math.max.apply(null, yCoords),
        leftSideBearing: 0
    };
    metrics.rightSideBearing = this.advanceWidth - metrics.leftSideBearing - (metrics.xMax - metrics.xMin);
    return metrics;
};

// Draw the glyph on the given context.
//
// ctx - The drawing context.
// x - Horizontal position of the glyph. (default: 0)
// y - Vertical position of the *baseline* of the glyph. (default: 0)
// fontSize - Font size, in pixels (default: 72).
Glyph.prototype.draw = function(ctx, x, y, fontSize) {
    this.getPath(x, y, fontSize).draw(ctx);
};

// Draw the points of the glyph.
// On-curve points will be drawn in blue, off-curve points will be drawn in red.
//
// ctx - The drawing context.
// x - Horizontal position of the glyph. (default: 0)
// y - Vertical position of the *baseline* of the glyph. (default: 0)
// fontSize - Font size, in pixels (default: 72).
Glyph.prototype.drawPoints = function(ctx, x, y, fontSize) {

    function drawCircles(l, x, y, scale) {
        var PI_SQ = Math.PI * 2;
        ctx.beginPath();
        for (var j = 0; j < l.length; j += 1) {
            ctx.moveTo(x + (l[j].x * scale), y + (l[j].y * scale));
            ctx.arc(x + (l[j].x * scale), y + (l[j].y * scale), 2, 0, PI_SQ, false);
        }

        ctx.closePath();
        ctx.fill();
    }

    x = x !== undefined ? x : 0;
    y = y !== undefined ? y : 0;
    fontSize = fontSize !== undefined ? fontSize : 24;
    var scale = 1 / this.path.unitsPerEm * fontSize;

    var blueCircles = [];
    var redCircles = [];
    var path = this.path;
    for (var i = 0; i < path.commands.length; i += 1) {
        var cmd = path.commands[i];
        if (cmd.x !== undefined) {
            blueCircles.push({x: cmd.x, y: -cmd.y});
        }

        if (cmd.x1 !== undefined) {
            redCircles.push({x: cmd.x1, y: -cmd.y1});
        }

        if (cmd.x2 !== undefined) {
            redCircles.push({x: cmd.x2, y: -cmd.y2});
        }
    }

    ctx.fillStyle = 'blue';
    drawCircles(blueCircles, x, y, scale);
    ctx.fillStyle = 'red';
    drawCircles(redCircles, x, y, scale);
};

// Draw lines indicating important font measurements.
// Black lines indicate the origin of the coordinate system (point 0,0).
// Blue lines indicate the glyph bounding box.
// Green line indicates the advance width of the glyph.
//
// ctx - The drawing context.
// x - Horizontal position of the glyph. (default: 0)
// y - Vertical position of the *baseline* of the glyph. (default: 0)
// fontSize - Font size, in pixels (default: 72).
Glyph.prototype.drawMetrics = function(ctx, x, y, fontSize) {
    var scale;
    x = x !== undefined ? x : 0;
    y = y !== undefined ? y : 0;
    fontSize = fontSize !== undefined ? fontSize : 24;
    scale = 1 / this.path.unitsPerEm * fontSize;
    ctx.lineWidth = 1;

    // Draw the origin
    ctx.strokeStyle = 'black';
    draw.line(ctx, x, -10000, x, 10000);
    draw.line(ctx, -10000, y, 10000, y);

    // This code is here due to memory optimization: by not using
    // defaults in the constructor, we save a notable amount of memory.
    var xMin = this.xMin || 0;
    var yMin = this.yMin || 0;
    var xMax = this.xMax || 0;
    var yMax = this.yMax || 0;
    var advanceWidth = this.advanceWidth || 0;

    // Draw the glyph box
    ctx.strokeStyle = 'blue';
    draw.line(ctx, x + (xMin * scale), -10000, x + (xMin * scale), 10000);
    draw.line(ctx, x + (xMax * scale), -10000, x + (xMax * scale), 10000);
    draw.line(ctx, -10000, y + (-yMin * scale), 10000, y + (-yMin * scale));
    draw.line(ctx, -10000, y + (-yMax * scale), 10000, y + (-yMax * scale));

    // Draw the advance width
    ctx.strokeStyle = 'green';
    draw.line(ctx, x + (advanceWidth * scale), -10000, x + (advanceWidth * scale), 10000);
};

exports.Glyph = Glyph;

},{"./check":2,"./draw":3,"./path":10}],7:[function(_dereq_,module,exports){
// The GlyphSet object

'use strict';

var _glyph = _dereq_('./glyph');

// A GlyphSet represents all glyphs available in the font, but modelled using
// a deferred glyph loader, for retrieving glyphs only once they are absolutely
// necessary, to keep the memory footprint down.
function GlyphSet(font, glyphs) {
    this.font = font;
    this.glyphs = {};
    if (Array.isArray(glyphs)) {
        for (var i = 0; i < glyphs.length; i++) {
            this.glyphs[i] = glyphs[i];
        }
    }

    this.length = (glyphs && glyphs.length) || 0;
}

GlyphSet.prototype.get = function(index) {
    if (typeof this.glyphs[index] === 'function') {
        this.glyphs[index] = this.glyphs[index]();
    }

    return this.glyphs[index];
};

GlyphSet.prototype.push = function(index, loader) {
    this.glyphs[index] = loader;
    this.length++;
};

function glyphLoader(font, index) {
    return new _glyph.Glyph({index: index, font: font});
}

/**
 * Generate a stub glyph that can be filled with all metadata *except*
 * the "points" and "path" properties, which must be loaded only once
 * the glyph's path is actually requested for text shaping.
 */

function ttfGlyphLoader(font, index, parseGlyph, data, position, buildPath) {
    return function() {
        var glyph = new _glyph.Glyph({index: index, font: font});

        glyph.path = function() {
            parseGlyph(glyph, data, position);
            var path = buildPath(font.glyphs, glyph);
            path.unitsPerEm = font.unitsPerEm;
            return path;
        };

        return glyph;
    };
}

function cffGlyphLoader(font, index, parseCFFCharstring, charstring) {
    return function() {
        var glyph = new _glyph.Glyph({index: index, font: font});

        glyph.path = function() {
            var path = parseCFFCharstring(font, glyph, charstring);
            path.unitsPerEm = font.unitsPerEm;
            return path;
        };

        return glyph;
    };
}

exports.GlyphSet = GlyphSet;
exports.glyphLoader = glyphLoader;
exports.ttfGlyphLoader = ttfGlyphLoader;
exports.cffGlyphLoader = cffGlyphLoader;

},{"./glyph":6}],8:[function(_dereq_,module,exports){
// opentype.js
// https://github.com/nodebox/opentype.js
// (c) 2015 Frederik De Bleser
// opentype.js may be freely distributed under the MIT license.

/* global ArrayBuffer, DataView, Uint8Array, XMLHttpRequest  */

'use strict';

var encoding = _dereq_('./encoding');
var _font = _dereq_('./font');
var glyph = _dereq_('./glyph');
var parse = _dereq_('./parse');
var path = _dereq_('./path');

var cmap = _dereq_('./tables/cmap');
var cff = _dereq_('./tables/cff');
var glyf = _dereq_('./tables/glyf');
var gpos = _dereq_('./tables/gpos');
var head = _dereq_('./tables/head');
var hhea = _dereq_('./tables/hhea');
var hmtx = _dereq_('./tables/hmtx');
var kern = _dereq_('./tables/kern');
var loca = _dereq_('./tables/loca');
var maxp = _dereq_('./tables/maxp');
var _name = _dereq_('./tables/name');
var os2 = _dereq_('./tables/os2');
var post = _dereq_('./tables/post');

// File loaders /////////////////////////////////////////////////////////

// Convert a Node.js Buffer to an ArrayBuffer
function toArrayBuffer(buffer) {
    var arrayBuffer = new ArrayBuffer(buffer.length);
    var data = new Uint8Array(arrayBuffer);
    for (var i = 0; i < buffer.length; i += 1) {
        data[i] = buffer[i];
    }

    return arrayBuffer;
}

function loadFromFile(path, callback) {
    var fs = _dereq_('fs');
    fs.readFile(path, function(err, buffer) {
        if (err) {
            return callback(err.message);
        }

        callback(null, toArrayBuffer(buffer));
    });
}

function loadFromUrl(url, callback) {
    var request = new XMLHttpRequest();
    request.open('get', url, true);
    request.responseType = 'arraybuffer';
    request.onload = function() {
        if (request.status !== 200) {
            return callback('Font could not be loaded: ' + request.statusText);
        }

        return callback(null, request.response);
    };

    request.send();
}

// Public API ///////////////////////////////////////////////////////////

// Parse the OpenType file data (as an ArrayBuffer) and return a Font object.
// If the file could not be parsed (most likely because it contains Postscript outlines)
// we return an empty Font object with the `supported` flag set to `false`.
function parseBuffer(buffer) {
    var indexToLocFormat;
    var hmtxOffset;
    var glyfOffset;
    var locaOffset;
    var cffOffset;
    var kernOffset;
    var gposOffset;

    // OpenType fonts use big endian byte ordering.
    // We can't rely on typed array view types, because they operate with the endianness of the host computer.
    // Instead we use DataViews where we can specify endianness.

    var font = new _font.Font();
    var data = new DataView(buffer, 0);

    var version = parse.getFixed(data, 0);
    if (version === 1.0) {
        font.outlinesFormat = 'truetype';
    } else {
        version = parse.getTag(data, 0);
        if (version === 'OTTO') {
            font.outlinesFormat = 'cff';
        } else {
            throw new Error('Unsupported OpenType version ' + version);
        }
    }

    var numTables = parse.getUShort(data, 4);

    // Offset into the table records.
    var p = 12;
    for (var i = 0; i < numTables; i += 1) {
        var tag = parse.getTag(data, p);
        var offset = parse.getULong(data, p + 8);
        switch (tag) {
        case 'cmap':
            font.tables.cmap = cmap.parse(data, offset);
            font.encoding = new encoding.CmapEncoding(font.tables.cmap);
            if (!font.encoding) {
                font.supported = false;
            }

            break;
        case 'head':
            font.tables.head = head.parse(data, offset);
            font.unitsPerEm = font.tables.head.unitsPerEm;
            indexToLocFormat = font.tables.head.indexToLocFormat;
            break;
        case 'hhea':
            font.tables.hhea = hhea.parse(data, offset);
            font.ascender = font.tables.hhea.ascender;
            font.descender = font.tables.hhea.descender;
            font.numberOfHMetrics = font.tables.hhea.numberOfHMetrics;
            break;
        case 'hmtx':
            hmtxOffset = offset;
            break;
        case 'maxp':
            font.tables.maxp = maxp.parse(data, offset);
            font.numGlyphs = font.tables.maxp.numGlyphs;
            break;
        case 'name':
            font.tables.name = _name.parse(data, offset);
            font.familyName = font.tables.name.fontFamily;
            font.styleName = font.tables.name.fontSubfamily;
            break;
        case 'OS/2':
            font.tables.os2 = os2.parse(data, offset);
            break;
        case 'post':
            font.tables.post = post.parse(data, offset);
            font.glyphNames = new encoding.GlyphNames(font.tables.post);
            break;
        case 'glyf':
            glyfOffset = offset;
            break;
        case 'loca':
            locaOffset = offset;
            break;
        case 'CFF ':
            cffOffset = offset;
            break;
        case 'kern':
            kernOffset = offset;
            break;
        case 'GPOS':
            gposOffset = offset;
            break;
        }
        p += 16;
    }

    if (glyfOffset && locaOffset) {
        var shortVersion = indexToLocFormat === 0;
        var locaTable = loca.parse(data, locaOffset, font.numGlyphs, shortVersion);
        font.glyphs = glyf.parse(data, glyfOffset, locaTable, font);
        hmtx.parse(data, hmtxOffset, font.numberOfHMetrics, font.numGlyphs, font.glyphs);
        encoding.addGlyphNames(font);
    } else if (cffOffset) {
        cff.parse(data, cffOffset, font);
        encoding.addGlyphNames(font);
    } else {
        font.supported = false;
    }

    if (font.supported) {
        if (kernOffset) {
            font.kerningPairs = kern.parse(data, kernOffset);
        } else {
            font.kerningPairs = {};
        }

        if (gposOffset) {
            gpos.parse(data, gposOffset, font);
        }
    }

    return font;
}

// Asynchronously load the font from a URL or a filesystem. When done, call the callback
// with two arguments `(err, font)`. The `err` will be null on success,
// the `font` is a Font object.
//
// We use the node.js callback convention so that
// opentype.js can integrate with frameworks like async.js.
function load(url, callback) {
    var isNode = typeof window === 'undefined';
    var loadFn = isNode ? loadFromFile : loadFromUrl;
    loadFn(url, function(err, arrayBuffer) {
        if (err) {
            return callback(err);
        }

        var font = parseBuffer(arrayBuffer);
        if (!font.supported) {
            return callback('Font is not supported (is this a Postscript font?)');
        }

        return callback(null, font);
    });
}

exports._parse = parse;
exports.Font = _font.Font;
exports.Glyph = glyph.Glyph;
exports.Path = path.Path;
exports.parse = parseBuffer;
exports.load = load;

},{"./encoding":4,"./font":5,"./glyph":6,"./parse":9,"./path":10,"./tables/cff":12,"./tables/cmap":13,"./tables/glyf":14,"./tables/gpos":15,"./tables/head":16,"./tables/hhea":17,"./tables/hmtx":18,"./tables/kern":19,"./tables/loca":20,"./tables/maxp":21,"./tables/name":22,"./tables/os2":23,"./tables/post":24,"fs":1}],9:[function(_dereq_,module,exports){
// Parsing utility functions

'use strict';

// Retrieve an unsigned byte from the DataView.
exports.getByte = function getByte(dataView, offset) {
    return dataView.getUint8(offset);
};

exports.getCard8 = exports.getByte;

// Retrieve an unsigned 16-bit short from the DataView.
// The value is stored in big endian.
exports.getUShort = function(dataView, offset) {
    return dataView.getUint16(offset, false);
};

exports.getCard16 = exports.getUShort;

// Retrieve a signed 16-bit short from the DataView.
// The value is stored in big endian.
exports.getShort = function(dataView, offset) {
    return dataView.getInt16(offset, false);
};

// Retrieve an unsigned 32-bit long from the DataView.
// The value is stored in big endian.
exports.getULong = function(dataView, offset) {
    return dataView.getUint32(offset, false);
};

// Retrieve a 32-bit signed fixed-point number (16.16) from the DataView.
// The value is stored in big endian.
exports.getFixed = function(dataView, offset) {
    var decimal = dataView.getInt16(offset, false);
    var fraction = dataView.getUint16(offset + 2, false);
    return decimal + fraction / 65535;
};

// Retrieve a 4-character tag from the DataView.
// Tags are used to identify tables.
exports.getTag = function(dataView, offset) {
    var tag = '';
    for (var i = offset; i < offset + 4; i += 1) {
        tag += String.fromCharCode(dataView.getInt8(i));
    }

    return tag;
};

// Retrieve an offset from the DataView.
// Offsets are 1 to 4 bytes in length, depending on the offSize argument.
exports.getOffset = function(dataView, offset, offSize) {
    var v = 0;
    for (var i = 0; i < offSize; i += 1) {
        v <<= 8;
        v += dataView.getUint8(offset + i);
    }

    return v;
};

// Retrieve a number of bytes from start offset to the end offset from the DataView.
exports.getBytes = function(dataView, startOffset, endOffset) {
    var bytes = [];
    for (var i = startOffset; i < endOffset; i += 1) {
        bytes.push(dataView.getUint8(i));
    }

    return bytes;
};

// Convert the list of bytes to a string.
exports.bytesToString = function(bytes) {
    var s = '';
    for (var i = 0; i < bytes.length; i += 1) {
        s += String.fromCharCode(bytes[i]);
    }

    return s;
};

var typeOffsets = {
    byte: 1,
    uShort: 2,
    short: 2,
    uLong: 4,
    fixed: 4,
    longDateTime: 8,
    tag: 4
};

// A stateful parser that changes the offset whenever a value is retrieved.
// The data is a DataView.
function Parser(data, offset) {
    this.data = data;
    this.offset = offset;
    this.relativeOffset = 0;
}

Parser.prototype.parseByte = function() {
    var v = this.data.getUint8(this.offset + this.relativeOffset);
    this.relativeOffset += 1;
    return v;
};

Parser.prototype.parseChar = function() {
    var v = this.data.getInt8(this.offset + this.relativeOffset);
    this.relativeOffset += 1;
    return v;
};

Parser.prototype.parseCard8 = Parser.prototype.parseByte;

Parser.prototype.parseUShort = function() {
    var v = this.data.getUint16(this.offset + this.relativeOffset);
    this.relativeOffset += 2;
    return v;
};

Parser.prototype.parseCard16 = Parser.prototype.parseUShort;
Parser.prototype.parseSID = Parser.prototype.parseUShort;
Parser.prototype.parseOffset16 = Parser.prototype.parseUShort;

Parser.prototype.parseShort = function() {
    var v = this.data.getInt16(this.offset + this.relativeOffset);
    this.relativeOffset += 2;
    return v;
};

Parser.prototype.parseF2Dot14 = function() {
    var v = this.data.getInt16(this.offset + this.relativeOffset) / 16384;
    this.relativeOffset += 2;
    return v;
};

Parser.prototype.parseULong = function() {
    var v = exports.getULong(this.data, this.offset + this.relativeOffset);
    this.relativeOffset += 4;
    return v;
};

Parser.prototype.parseFixed = function() {
    var v = exports.getFixed(this.data, this.offset + this.relativeOffset);
    this.relativeOffset += 4;
    return v;
};

Parser.prototype.parseOffset16List =
Parser.prototype.parseUShortList = function(count) {
    var offsets = new Array(count);
    var dataView = this.data;
    var offset = this.offset + this.relativeOffset;
    for (var i = 0; i < count; i++) {
        offsets[i] = exports.getUShort(dataView, offset);
        offset += 2;
    }

    this.relativeOffset += count * 2;
    return offsets;
};

Parser.prototype.parseString = function(length) {
    var dataView = this.data;
    var offset = this.offset + this.relativeOffset;
    var string = '';
    this.relativeOffset += length;
    for (var i = 0; i < length; i++) {
        string += String.fromCharCode(dataView.getUint8(offset + i));
    }

    return string;
};

Parser.prototype.parseTag = function() {
    return this.parseString(4);
};

// LONGDATETIME is a 64-bit integer.
// JavaScript and unix timestamps traditionally use 32 bits, so we
// only take the last 32 bits.
Parser.prototype.parseLongDateTime = function() {
    var v = exports.getULong(this.data, this.offset + this.relativeOffset + 4);
    this.relativeOffset += 8;
    return v;
};

Parser.prototype.parseFixed = function() {
    var v = exports.getULong(this.data, this.offset + this.relativeOffset);
    this.relativeOffset += 4;
    return v / 65536;
};

Parser.prototype.parseVersion = function() {
    var major = exports.getUShort(this.data, this.offset + this.relativeOffset);

    // How to interpret the minor version is very vague in the spec. 0x5000 is 5, 0x1000 is 1
    // This returns the correct number if minor = 0xN000 where N is 0-9
    var minor = exports.getUShort(this.data, this.offset + this.relativeOffset + 2);
    this.relativeOffset += 4;
    return major + minor / 0x1000 / 10;
};

Parser.prototype.skip = function(type, amount) {
    if (amount === undefined) {
        amount = 1;
    }

    this.relativeOffset += typeOffsets[type] * amount;
};

exports.Parser = Parser;

},{}],10:[function(_dereq_,module,exports){
// Geometric objects

'use strict';

// A bézier path containing a set of path commands similar to a SVG path.
// Paths can be drawn on a context using `draw`.
function Path() {
    this.commands = [];
    this.fill = 'black';
    this.stroke = null;
    this.strokeWidth = 1;
}

Path.prototype.moveTo = function(x, y) {
    this.commands.push({
        type: 'M',
        x: x,
        y: y
    });
};

Path.prototype.lineTo = function(x, y) {
    this.commands.push({
        type: 'L',
        x: x,
        y: y
    });
};

Path.prototype.curveTo = Path.prototype.bezierCurveTo = function(x1, y1, x2, y2, x, y) {
    this.commands.push({
        type: 'C',
        x1: x1,
        y1: y1,
        x2: x2,
        y2: y2,
        x: x,
        y: y
    });
};

Path.prototype.quadTo = Path.prototype.quadraticCurveTo = function(x1, y1, x, y) {
    this.commands.push({
        type: 'Q',
        x1: x1,
        y1: y1,
        x: x,
        y: y
    });
};

Path.prototype.close = Path.prototype.closePath = function() {
    this.commands.push({
        type: 'Z'
    });
};

// Add the given path or list of commands to the commands of this path.
Path.prototype.extend = function(pathOrCommands) {
    if (pathOrCommands.commands) {
        pathOrCommands = pathOrCommands.commands;
    }

    Array.prototype.push.apply(this.commands, pathOrCommands);
};

// Draw the path to a 2D context.
Path.prototype.draw = function(ctx) {
    ctx.beginPath();
    for (var i = 0; i < this.commands.length; i += 1) {
        var cmd = this.commands[i];
        if (cmd.type === 'M') {
            ctx.moveTo(cmd.x, cmd.y);
        } else if (cmd.type === 'L') {
            ctx.lineTo(cmd.x, cmd.y);
        } else if (cmd.type === 'C') {
            ctx.bezierCurveTo(cmd.x1, cmd.y1, cmd.x2, cmd.y2, cmd.x, cmd.y);
        } else if (cmd.type === 'Q') {
            ctx.quadraticCurveTo(cmd.x1, cmd.y1, cmd.x, cmd.y);
        } else if (cmd.type === 'Z') {
            ctx.closePath();
        }
    }

    if (this.fill) {
        ctx.fillStyle = this.fill;
        ctx.fill();
    }

    if (this.stroke) {
        ctx.strokeStyle = this.stroke;
        ctx.lineWidth = this.strokeWidth;
        ctx.stroke();
    }
};

// Convert the Path to a string of path data instructions
// See http://www.w3.org/TR/SVG/paths.html#PathData
// Parameters:
// - decimalPlaces: The amount of decimal places for floating-point values (default: 2)
Path.prototype.toPathData = function(decimalPlaces) {
    decimalPlaces = decimalPlaces !== undefined ? decimalPlaces : 2;

    function floatToString(v) {
        if (Math.round(v) === v) {
            return '' + Math.round(v);
        } else {
            return v.toFixed(decimalPlaces);
        }
    }

    function packValues() {
        var s = '';
        for (var i = 0; i < arguments.length; i += 1) {
            var v = arguments[i];
            if (v >= 0 && i > 0) {
                s += ' ';
            }

            s += floatToString(v);
        }

        return s;
    }

    var d = '';
    for (var i = 0; i < this.commands.length; i += 1) {
        var cmd = this.commands[i];
        if (cmd.type === 'M') {
            d += 'M' + packValues(cmd.x, cmd.y);
        } else if (cmd.type === 'L') {
            d += 'L' + packValues(cmd.x, cmd.y);
        } else if (cmd.type === 'C') {
            d += 'C' + packValues(cmd.x1, cmd.y1, cmd.x2, cmd.y2, cmd.x, cmd.y);
        } else if (cmd.type === 'Q') {
            d += 'Q' + packValues(cmd.x1, cmd.y1, cmd.x, cmd.y);
        } else if (cmd.type === 'Z') {
            d += 'Z';
        }
    }

    return d;
};

// Convert the path to a SVG <path> element, as a string.
// Parameters:
// - decimalPlaces: The amount of decimal places for floating-point values (default: 2)
Path.prototype.toSVG = function(decimalPlaces) {
    var svg = '<path d="';
    svg += this.toPathData(decimalPlaces);
    svg += '"';
    if (this.fill & this.fill !== 'black') {
        if (this.fill === null) {
            svg += ' fill="none"';
        } else {
            svg += ' fill="' + this.fill + '"';
        }
    }

    if (this.stroke) {
        svg += ' stroke="' + this.stroke + '" stroke-width="' + this.strokeWidth + '"';
    }

    svg += '/>';
    return svg;
};

exports.Path = Path;

},{}],11:[function(_dereq_,module,exports){
// Table metadata

'use strict';

var check = _dereq_('./check');
var encode = _dereq_('./types').encode;
var sizeOf = _dereq_('./types').sizeOf;

function Table(tableName, fields, options) {
    var i;
    for (i = 0; i < fields.length; i += 1) {
        var field = fields[i];
        this[field.name] = field.value;
    }

    this.tableName = tableName;
    this.fields = fields;
    if (options) {
        var optionKeys = Object.keys(options);
        for (i = 0; i < optionKeys.length; i += 1) {
            var k = optionKeys[i];
            var v = options[k];
            if (this[k] !== undefined) {
                this[k] = v;
            }
        }
    }
}

Table.prototype.sizeOf = function() {
    var v = 0;
    for (var i = 0; i < this.fields.length; i += 1) {
        var field = this.fields[i];
        var value = this[field.name];
        if (value === undefined) {
            value = field.value;
        }

        if (typeof value.sizeOf === 'function') {
            v += value.sizeOf();
        } else {
            var sizeOfFunction = sizeOf[field.type];
            check.assert(typeof sizeOfFunction === 'function', 'Could not find sizeOf function for field' + field.name);
            v += sizeOfFunction(value);
        }
    }

    return v;
};

Table.prototype.encode = function() {
    return encode.TABLE(this);
};

exports.Table = Table;

},{"./check":2,"./types":26}],12:[function(_dereq_,module,exports){
// The `CFF` table contains the glyph outlines in PostScript format.
// https://www.microsoft.com/typography/OTSPEC/cff.htm
// http://download.microsoft.com/download/8/0/1/801a191c-029d-4af3-9642-555f6fe514ee/cff.pdf
// http://download.microsoft.com/download/8/0/1/801a191c-029d-4af3-9642-555f6fe514ee/type2.pdf

'use strict';

var encoding = _dereq_('../encoding');
var glyphset = _dereq_('../glyphset');
var parse = _dereq_('../parse');
var path = _dereq_('../path');
var table = _dereq_('../table');

// Custom equals function that can also check lists.
function equals(a, b) {
    if (a === b) {
        return true;
    } else if (Array.isArray(a) && Array.isArray(b)) {
        if (a.length !== b.length) {
            return false;
        }

        for (var i = 0; i < a.length; i += 1) {
            if (!equals(a[i], b[i])) {
                return false;
            }
        }

        return true;
    } else {
        return false;
    }
}

// Parse a `CFF` INDEX array.
// An index array consists of a list of offsets, then a list of objects at those offsets.
function parseCFFIndex(data, start, conversionFn) {
    //var i, objectOffset, endOffset;
    var offsets = [];
    var objects = [];
    var count = parse.getCard16(data, start);
    var i;
    var objectOffset;
    var endOffset;
    if (count !== 0) {
        var offsetSize = parse.getByte(data, start + 2);
        objectOffset = start + ((count + 1) * offsetSize) + 2;
        var pos = start + 3;
        for (i = 0; i < count + 1; i += 1) {
            offsets.push(parse.getOffset(data, pos, offsetSize));
            pos += offsetSize;
        }

        // The total size of the index array is 4 header bytes + the value of the last offset.
        endOffset = objectOffset + offsets[count];
    } else {
        endOffset = start + 2;
    }

    for (i = 0; i < offsets.length - 1; i += 1) {
        var value = parse.getBytes(data, objectOffset + offsets[i], objectOffset + offsets[i + 1]);
        if (conversionFn) {
            value = conversionFn(value);
        }

        objects.push(value);
    }

    return {objects: objects, startOffset: start, endOffset: endOffset};
}

// Parse a `CFF` DICT real value.
function parseFloatOperand(parser) {
    var s = '';
    var eof = 15;
    var lookup = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '.', 'E', 'E-', null, '-'];
    while (true) {
        var b = parser.parseByte();
        var n1 = b >> 4;
        var n2 = b & 15;

        if (n1 === eof) {
            break;
        }

        s += lookup[n1];

        if (n2 === eof) {
            break;
        }

        s += lookup[n2];
    }

    return parseFloat(s);
}

// Parse a `CFF` DICT operand.
function parseOperand(parser, b0) {
    var b1;
    var b2;
    var b3;
    var b4;
    if (b0 === 28) {
        b1 = parser.parseByte();
        b2 = parser.parseByte();
        return b1 << 8 | b2;
    }

    if (b0 === 29) {
        b1 = parser.parseByte();
        b2 = parser.parseByte();
        b3 = parser.parseByte();
        b4 = parser.parseByte();
        return b1 << 24 | b2 << 16 | b3 << 8 | b4;
    }

    if (b0 === 30) {
        return parseFloatOperand(parser);
    }

    if (b0 >= 32 && b0 <= 246) {
        return b0 - 139;
    }

    if (b0 >= 247 && b0 <= 250) {
        b1 = parser.parseByte();
        return (b0 - 247) * 256 + b1 + 108;
    }

    if (b0 >= 251 && b0 <= 254) {
        b1 = parser.parseByte();
        return -(b0 - 251) * 256 - b1 - 108;
    }

    throw new Error('Invalid b0 ' + b0);
}

// Convert the entries returned by `parseDict` to a proper dictionary.
// If a value is a list of one, it is unpacked.
function entriesToObject(entries) {
    var o = {};
    for (var i = 0; i < entries.length; i += 1) {
        var key = entries[i][0];
        var values = entries[i][1];
        var value;
        if (values.length === 1) {
            value = values[0];
        } else {
            value = values;
        }

        if (o.hasOwnProperty(key)) {
            throw new Error('Object ' + o + ' already has key ' + key);
        }

        o[key] = value;
    }

    return o;
}

// Parse a `CFF` DICT object.
// A dictionary contains key-value pairs in a compact tokenized format.
function parseCFFDict(data, start, size) {
    start = start !== undefined ? start : 0;
    var parser = new parse.Parser(data, start);
    var entries = [];
    var operands = [];
    size = size !== undefined ? size : data.length;

    while (parser.relativeOffset < size) {
        var op = parser.parseByte();

        // The first byte for each dict item distinguishes between operator (key) and operand (value).
        // Values <= 21 are operators.
        if (op <= 21) {
            // Two-byte operators have an initial escape byte of 12.
            if (op === 12) {
                op = 1200 + parser.parseByte();
            }

            entries.push([op, operands]);
            operands = [];
        } else {
            // Since the operands (values) come before the operators (keys), we store all operands in a list
            // until we encounter an operator.
            operands.push(parseOperand(parser, op));
        }
    }

    return entriesToObject(entries);
}

// Given a String Index (SID), return the value of the string.
// Strings below index 392 are standard CFF strings and are not encoded in the font.
function getCFFString(strings, index) {
    if (index <= 390) {
        index = encoding.cffStandardStrings[index];
    } else {
        index = strings[index - 391];
    }

    return index;
}

// Interpret a dictionary and return a new dictionary with readable keys and values for missing entries.
// This function takes `meta` which is a list of objects containing `operand`, `name` and `default`.
function interpretDict(dict, meta, strings) {
    var newDict = {};

    // Because we also want to include missing values, we start out from the meta list
    // and lookup values in the dict.
    for (var i = 0; i < meta.length; i += 1) {
        var m = meta[i];
        var value = dict[m.op];
        if (value === undefined) {
            value = m.value !== undefined ? m.value : null;
        }

        if (m.type === 'SID') {
            value = getCFFString(strings, value);
        }

        newDict[m.name] = value;
    }

    return newDict;
}

// Parse the CFF header.
function parseCFFHeader(data, start) {
    var header = {};
    header.formatMajor = parse.getCard8(data, start);
    header.formatMinor = parse.getCard8(data, start + 1);
    header.size = parse.getCard8(data, start + 2);
    header.offsetSize = parse.getCard8(data, start + 3);
    header.startOffset = start;
    header.endOffset = start + 4;
    return header;
}

var TOP_DICT_META = [
    {name: 'version', op: 0, type: 'SID'},
    {name: 'notice', op: 1, type: 'SID'},
    {name: 'copyright', op: 1200, type: 'SID'},
    {name: 'fullName', op: 2, type: 'SID'},
    {name: 'familyName', op: 3, type: 'SID'},
    {name: 'weight', op: 4, type: 'SID'},
    {name: 'isFixedPitch', op: 1201, type: 'number', value: 0},
    {name: 'italicAngle', op: 1202, type: 'number', value: 0},
    {name: 'underlinePosition', op: 1203, type: 'number', value: -100},
    {name: 'underlineThickness', op: 1204, type: 'number', value: 50},
    {name: 'paintType', op: 1205, type: 'number', value: 0},
    {name: 'charstringType', op: 1206, type: 'number', value: 2},
    {name: 'fontMatrix', op: 1207, type: ['real', 'real', 'real', 'real', 'real', 'real'], value: [0.001, 0, 0, 0.001, 0, 0]},
    {name: 'uniqueId', op: 13, type: 'number'},
    {name: 'fontBBox', op: 5, type: ['number', 'number', 'number', 'number'], value: [0, 0, 0, 0]},
    {name: 'strokeWidth', op: 1208, type: 'number', value: 0},
    {name: 'xuid', op: 14, type: [], value: null},
    {name: 'charset', op: 15, type: 'offset', value: 0},
    {name: 'encoding', op: 16, type: 'offset', value: 0},
    {name: 'charStrings', op: 17, type: 'offset', value: 0},
    {name: 'private', op: 18, type: ['number', 'offset'], value: [0, 0]}
];

var PRIVATE_DICT_META = [
    {name: 'subrs', op: 19, type: 'offset', value: 0},
    {name: 'defaultWidthX', op: 20, type: 'number', value: 0},
    {name: 'nominalWidthX', op: 21, type: 'number', value: 0}
];

// Parse the CFF top dictionary. A CFF table can contain multiple fonts, each with their own top dictionary.
// The top dictionary contains the essential metadata for the font, together with the private dictionary.
function parseCFFTopDict(data, strings) {
    var dict = parseCFFDict(data, 0, data.byteLength);
    return interpretDict(dict, TOP_DICT_META, strings);
}

// Parse the CFF private dictionary. We don't fully parse out all the values, only the ones we need.
function parseCFFPrivateDict(data, start, size, strings) {
    var dict = parseCFFDict(data, start, size);
    return interpretDict(dict, PRIVATE_DICT_META, strings);
}

// Parse the CFF charset table, which contains internal names for all the glyphs.
// This function will return a list of glyph names.
// See Adobe TN #5176 chapter 13, "Charsets".
function parseCFFCharset(data, start, nGlyphs, strings) {
    var i;
    var sid;
    var count;
    var parser = new parse.Parser(data, start);

    // The .notdef glyph is not included, so subtract 1.
    nGlyphs -= 1;
    var charset = ['.notdef'];

    var format = parser.parseCard8();
    if (format === 0) {
        for (i = 0; i < nGlyphs; i += 1) {
            sid = parser.parseSID();
            charset.push(getCFFString(strings, sid));
        }
    } else if (format === 1) {
        while (charset.length <= nGlyphs) {
            sid = parser.parseSID();
            count = parser.parseCard8();
            for (i = 0; i <= count; i += 1) {
                charset.push(getCFFString(strings, sid));
                sid += 1;
            }
        }
    } else if (format === 2) {
        while (charset.length <= nGlyphs) {
            sid = parser.parseSID();
            count = parser.parseCard16();
            for (i = 0; i <= count; i += 1) {
                charset.push(getCFFString(strings, sid));
                sid += 1;
            }
        }
    } else {
        throw new Error('Unknown charset format ' + format);
    }

    return charset;
}

// Parse the CFF encoding data. Only one encoding can be specified per font.
// See Adobe TN #5176 chapter 12, "Encodings".
function parseCFFEncoding(data, start, charset) {
    var i;
    var code;
    var enc = {};
    var parser = new parse.Parser(data, start);
    var format = parser.parseCard8();
    if (format === 0) {
        var nCodes = parser.parseCard8();
        for (i = 0; i < nCodes; i += 1) {
            code = parser.parseCard8();
            enc[code] = i;
        }
    } else if (format === 1) {
        var nRanges = parser.parseCard8();
        code = 1;
        for (i = 0; i < nRanges; i += 1) {
            var first = parser.parseCard8();
            var nLeft = parser.parseCard8();
            for (var j = first; j <= first + nLeft; j += 1) {
                enc[j] = code;
                code += 1;
            }
        }
    } else {
        throw new Error('Unknown encoding format ' + format);
    }

    return new encoding.CffEncoding(enc, charset);
}

// Take in charstring code and return a Glyph object.
// The encoding is described in the Type 2 Charstring Format
// https://www.microsoft.com/typography/OTSPEC/charstr2.htm
function parseCFFCharstring(font, glyph, code) {
    var c1x;
    var c1y;
    var c2x;
    var c2y;
    var p = new path.Path();
    var stack = [];
    var nStems = 0;
    var haveWidth = false;
    var width = font.defaultWidthX;
    var open = false;
    var x = 0;
    var y = 0;

    function newContour(x, y) {
        if (open) {
            p.closePath();
        }

        p.moveTo(x, y);
        open = true;
    }

    function parseStems() {
        var hasWidthArg;

        // The number of stem operators on the stack is always even.
        // If the value is uneven, that means a width is specified.
        hasWidthArg = stack.length % 2 !== 0;
        if (hasWidthArg && !haveWidth) {
            width = stack.shift() + font.nominalWidthX;
        }

        nStems += stack.length >> 1;
        stack.length = 0;
        haveWidth = true;
    }

    function parse(code) {
        var b1;
        var b2;
        var b3;
        var b4;
        var codeIndex;
        var subrCode;
        var jpx;
        var jpy;
        var c3x;
        var c3y;
        var c4x;
        var c4y;

        var i = 0;
        while (i < code.length) {
            var v = code[i];
            i += 1;
            switch (v) {
            case 1: // hstem
                parseStems();
                break;
            case 3: // vstem
                parseStems();
                break;
            case 4: // vmoveto
                if (stack.length > 1 && !haveWidth) {
                    width = stack.shift() + font.nominalWidthX;
                    haveWidth = true;
                }

                y += stack.pop();
                newContour(x, y);
                break;
            case 5: // rlineto
                while (stack.length > 0) {
                    x += stack.shift();
                    y += stack.shift();
                    p.lineTo(x, y);
                }

                break;
            case 6: // hlineto
                while (stack.length > 0) {
                    x += stack.shift();
                    p.lineTo(x, y);
                    if (stack.length === 0) {
                        break;
                    }

                    y += stack.shift();
                    p.lineTo(x, y);
                }

                break;
            case 7: // vlineto
                while (stack.length > 0) {
                    y += stack.shift();
                    p.lineTo(x, y);
                    if (stack.length === 0) {
                        break;
                    }

                    x += stack.shift();
                    p.lineTo(x, y);
                }

                break;
            case 8: // rrcurveto
                while (stack.length > 0) {
                    c1x = x + stack.shift();
                    c1y = y + stack.shift();
                    c2x = c1x + stack.shift();
                    c2y = c1y + stack.shift();
                    x = c2x + stack.shift();
                    y = c2y + stack.shift();
                    p.curveTo(c1x, c1y, c2x, c2y, x, y);
                }

                break;
            case 10: // callsubr
                codeIndex = stack.pop() + font.subrsBias;
                subrCode = font.subrs[codeIndex];
                if (subrCode) {
                    parse(subrCode);
                }

                break;
            case 11: // return
                return;
            case 12: // flex operators
                v = code[i];
                i += 1;
                switch (v) {
                case 35: // flex
                    // |- dx1 dy1 dx2 dy2 dx3 dy3 dx4 dy4 dx5 dy5 dx6 dy6 fd flex (12 35) |-
                    c1x = x   + stack.shift();    // dx1
                    c1y = y   + stack.shift();    // dy1
                    c2x = c1x + stack.shift();    // dx2
                    c2y = c1y + stack.shift();    // dy2
                    jpx = c2x + stack.shift();    // dx3
                    jpy = c2y + stack.shift();    // dy3
                    c3x = jpx + stack.shift();    // dx4
                    c3y = jpy + stack.shift();    // dy4
                    c4x = c3x + stack.shift();    // dx5
                    c4y = c3y + stack.shift();    // dy5
                    x = c4x + stack.shift();      // dx6
                    y = c4y + stack.shift();      // dy6
                    stack.shift();                // flex depth
                    p.curveTo(c1x, c1y, c2x, c2y, jpx, jpy);
                    p.curveTo(c3x, c3y, c4x, c4y, x, y);
                    break;
                case 34: // hflex
                    // |- dx1 dx2 dy2 dx3 dx4 dx5 dx6 hflex (12 34) |-
                    c1x = x   + stack.shift();    // dx1
                    c1y = y;                      // dy1
                    c2x = c1x + stack.shift();    // dx2
                    c2y = c1y + stack.shift();    // dy2
                    jpx = c2x + stack.shift();    // dx3
                    jpy = c2y;                    // dy3
                    c3x = jpx + stack.shift();    // dx4
                    c3y = c2y;                    // dy4
                    c4x = c3x + stack.shift();    // dx5
                    c4y = y;                      // dy5
                    x = c4x + stack.shift();      // dx6
                    p.curveTo(c1x, c1y, c2x, c2y, jpx, jpy);
                    p.curveTo(c3x, c3y, c4x, c4y, x, y);
                    break;
                case 36: // hflex1
                    // |- dx1 dy1 dx2 dy2 dx3 dx4 dx5 dy5 dx6 hflex1 (12 36) |-
                    c1x = x   + stack.shift();    // dx1
                    c1y = y   + stack.shift();    // dy1
                    c2x = c1x + stack.shift();    // dx2
                    c2y = c1y + stack.shift();    // dy2
                    jpx = c2x + stack.shift();    // dx3
                    jpy = c2y;                    // dy3
                    c3x = jpx + stack.shift();    // dx4
                    c3y = c2y;                    // dy4
                    c4x = c3x + stack.shift();    // dx5
                    c4y = c3y + stack.shift();    // dy5
                    x = c4x + stack.shift();      // dx6
                    p.curveTo(c1x, c1y, c2x, c2y, jpx, jpy);
                    p.curveTo(c3x, c3y, c4x, c4y, x, y);
                    break;
                case 37: // flex1
                    // |- dx1 dy1 dx2 dy2 dx3 dy3 dx4 dy4 dx5 dy5 d6 flex1 (12 37) |-
                    c1x = x   + stack.shift();    // dx1
                    c1y = y   + stack.shift();    // dy1
                    c2x = c1x + stack.shift();    // dx2
                    c2y = c1y + stack.shift();    // dy2
                    jpx = c2x + stack.shift();    // dx3
                    jpy = c2y + stack.shift();    // dy3
                    c3x = jpx + stack.shift();    // dx4
                    c3y = jpy + stack.shift();    // dy4
                    c4x = c3x + stack.shift();    // dx5
                    c4y = c3y + stack.shift();    // dy5
                    if (Math.abs(c4x - x) > Math.abs(c4y - y)) {
                        x = c4x + stack.shift();
                    } else {
                        y = c4y + stack.shift();
                    }

                    p.curveTo(c1x, c1y, c2x, c2y, jpx, jpy);
                    p.curveTo(c3x, c3y, c4x, c4y, x, y);
                    break;
                default:
                    console.log('Glyph ' + glyph.index + ': unknown operator ' + 1200 + v);
                    stack.length = 0;
                }
                break;
            case 14: // endchar
                if (stack.length > 0 && !haveWidth) {
                    width = stack.shift() + font.nominalWidthX;
                    haveWidth = true;
                }

                if (open) {
                    p.closePath();
                    open = false;
                }

                break;
            case 18: // hstemhm
                parseStems();
                break;
            case 19: // hintmask
            case 20: // cntrmask
                parseStems();
                i += (nStems + 7) >> 3;
                break;
            case 21: // rmoveto
                if (stack.length > 2 && !haveWidth) {
                    width = stack.shift() + font.nominalWidthX;
                    haveWidth = true;
                }

                y += stack.pop();
                x += stack.pop();
                newContour(x, y);
                break;
            case 22: // hmoveto
                if (stack.length > 1 && !haveWidth) {
                    width = stack.shift() + font.nominalWidthX;
                    haveWidth = true;
                }

                x += stack.pop();
                newContour(x, y);
                break;
            case 23: // vstemhm
                parseStems();
                break;
            case 24: // rcurveline
                while (stack.length > 2) {
                    c1x = x + stack.shift();
                    c1y = y + stack.shift();
                    c2x = c1x + stack.shift();
                    c2y = c1y + stack.shift();
                    x = c2x + stack.shift();
                    y = c2y + stack.shift();
                    p.curveTo(c1x, c1y, c2x, c2y, x, y);
                }

                x += stack.shift();
                y += stack.shift();
                p.lineTo(x, y);
                break;
            case 25: // rlinecurve
                while (stack.length > 6) {
                    x += stack.shift();
                    y += stack.shift();
                    p.lineTo(x, y);
                }

                c1x = x + stack.shift();
                c1y = y + stack.shift();
                c2x = c1x + stack.shift();
                c2y = c1y + stack.shift();
                x = c2x + stack.shift();
                y = c2y + stack.shift();
                p.curveTo(c1x, c1y, c2x, c2y, x, y);
                break;
            case 26: // vvcurveto
                if (stack.length % 2) {
                    x += stack.shift();
                }

                while (stack.length > 0) {
                    c1x = x;
                    c1y = y + stack.shift();
                    c2x = c1x + stack.shift();
                    c2y = c1y + stack.shift();
                    x = c2x;
                    y = c2y + stack.shift();
                    p.curveTo(c1x, c1y, c2x, c2y, x, y);
                }

                break;
            case 27: // hhcurveto
                if (stack.length % 2) {
                    y += stack.shift();
                }

                while (stack.length > 0) {
                    c1x = x + stack.shift();
                    c1y = y;
                    c2x = c1x + stack.shift();
                    c2y = c1y + stack.shift();
                    x = c2x + stack.shift();
                    y = c2y;
                    p.curveTo(c1x, c1y, c2x, c2y, x, y);
                }

                break;
            case 28: // shortint
                b1 = code[i];
                b2 = code[i + 1];
                stack.push(((b1 << 24) | (b2 << 16)) >> 16);
                i += 2;
                break;
            case 29: // callgsubr
                codeIndex = stack.pop() + font.gsubrsBias;
                subrCode = font.gsubrs[codeIndex];
                if (subrCode) {
                    parse(subrCode);
                }

                break;
            case 30: // vhcurveto
                while (stack.length > 0) {
                    c1x = x;
                    c1y = y + stack.shift();
                    c2x = c1x + stack.shift();
                    c2y = c1y + stack.shift();
                    x = c2x + stack.shift();
                    y = c2y + (stack.length === 1 ? stack.shift() : 0);
                    p.curveTo(c1x, c1y, c2x, c2y, x, y);
                    if (stack.length === 0) {
                        break;
                    }

                    c1x = x + stack.shift();
                    c1y = y;
                    c2x = c1x + stack.shift();
                    c2y = c1y + stack.shift();
                    y = c2y + stack.shift();
                    x = c2x + (stack.length === 1 ? stack.shift() : 0);
                    p.curveTo(c1x, c1y, c2x, c2y, x, y);
                }

                break;
            case 31: // hvcurveto
                while (stack.length > 0) {
                    c1x = x + stack.shift();
                    c1y = y;
                    c2x = c1x + stack.shift();
                    c2y = c1y + stack.shift();
                    y = c2y + stack.shift();
                    x = c2x + (stack.length === 1 ? stack.shift() : 0);
                    p.curveTo(c1x, c1y, c2x, c2y, x, y);
                    if (stack.length === 0) {
                        break;
                    }

                    c1x = x;
                    c1y = y + stack.shift();
                    c2x = c1x + stack.shift();
                    c2y = c1y + stack.shift();
                    x = c2x + stack.shift();
                    y = c2y + (stack.length === 1 ? stack.shift() : 0);
                    p.curveTo(c1x, c1y, c2x, c2y, x, y);
                }

                break;
            default:
                if (v < 32) {
                    console.log('Glyph ' + glyph.index + ': unknown operator ' + v);
                } else if (v < 247) {
                    stack.push(v - 139);
                } else if (v < 251) {
                    b1 = code[i];
                    i += 1;
                    stack.push((v - 247) * 256 + b1 + 108);
                } else if (v < 255) {
                    b1 = code[i];
                    i += 1;
                    stack.push(-(v - 251) * 256 - b1 - 108);
                } else {
                    b1 = code[i];
                    b2 = code[i + 1];
                    b3 = code[i + 2];
                    b4 = code[i + 3];
                    i += 4;
                    stack.push(((b1 << 24) | (b2 << 16) | (b3 << 8) | b4) / 65536);
                }
            }
        }
    }

    parse(code);

    glyph.advanceWidth = width;
    return p;
}

// Subroutines are encoded using the negative half of the number space.
// See type 2 chapter 4.7 "Subroutine operators".
function calcCFFSubroutineBias(subrs) {
    var bias;
    if (subrs.length < 1240) {
        bias = 107;
    } else if (subrs.length < 33900) {
        bias = 1131;
    } else {
        bias = 32768;
    }

    return bias;
}

// Parse the `CFF` table, which contains the glyph outlines in PostScript format.
function parseCFFTable(data, start, font) {
    font.tables.cff = {};
    var header = parseCFFHeader(data, start);
    var nameIndex = parseCFFIndex(data, header.endOffset, parse.bytesToString);
    var topDictIndex = parseCFFIndex(data, nameIndex.endOffset);
    var stringIndex = parseCFFIndex(data, topDictIndex.endOffset, parse.bytesToString);
    var globalSubrIndex = parseCFFIndex(data, stringIndex.endOffset);
    font.gsubrs = globalSubrIndex.objects;
    font.gsubrsBias = calcCFFSubroutineBias(font.gsubrs);

    var topDictData = new DataView(new Uint8Array(topDictIndex.objects[0]).buffer);
    var topDict = parseCFFTopDict(topDictData, stringIndex.objects);
    font.tables.cff.topDict = topDict;

    var privateDictOffset = start + topDict['private'][1];
    var privateDict = parseCFFPrivateDict(data, privateDictOffset, topDict['private'][0], stringIndex.objects);
    font.defaultWidthX = privateDict.defaultWidthX;
    font.nominalWidthX = privateDict.nominalWidthX;

    if (privateDict.subrs !== 0) {
        var subrOffset = privateDictOffset + privateDict.subrs;
        var subrIndex = parseCFFIndex(data, subrOffset);
        font.subrs = subrIndex.objects;
        font.subrsBias = calcCFFSubroutineBias(font.subrs);
    } else {
        font.subrs = [];
        font.subrsBias = 0;
    }

    // Offsets in the top dict are relative to the beginning of the CFF data, so add the CFF start offset.
    var charStringsIndex = parseCFFIndex(data, start + topDict.charStrings);
    font.nGlyphs = charStringsIndex.objects.length;

    var charset = parseCFFCharset(data, start + topDict.charset, font.nGlyphs, stringIndex.objects);
    if (topDict.encoding === 0) { // Standard encoding
        font.cffEncoding = new encoding.CffEncoding(encoding.cffStandardEncoding, charset);
    } else if (topDict.encoding === 1) { // Expert encoding
        font.cffEncoding = new encoding.CffEncoding(encoding.cffExpertEncoding, charset);
    } else {
        font.cffEncoding = parseCFFEncoding(data, start + topDict.encoding, charset);
    }

    // Prefer the CMAP encoding to the CFF encoding.
    font.encoding = font.encoding || font.cffEncoding;

    font.glyphs = new glyphset.GlyphSet(font);
    for (var i = 0; i < font.nGlyphs; i += 1) {
        var charString = charStringsIndex.objects[i];
        font.glyphs.push(i, glyphset.cffGlyphLoader(font, i, parseCFFCharstring, charString));
    }
}

// Convert a string to a String ID (SID).
// The list of strings is modified in place.
function encodeString(s, strings) {
    var sid;

    // Is the string in the CFF standard strings?
    var i = encoding.cffStandardStrings.indexOf(s);
    if (i >= 0) {
        sid = i;
    }

    // Is the string already in the string index?
    i = strings.indexOf(s);
    if (i >= 0) {
        sid = i + encoding.cffStandardStrings.length;
    } else {
        sid = encoding.cffStandardStrings.length + strings.length;
        strings.push(s);
    }

    return sid;
}

function makeHeader() {
    return new table.Table('Header', [
        {name: 'major', type: 'Card8', value: 1},
        {name: 'minor', type: 'Card8', value: 0},
        {name: 'hdrSize', type: 'Card8', value: 4},
        {name: 'major', type: 'Card8', value: 1}
    ]);
}

function makeNameIndex(fontNames) {
    var t = new table.Table('Name INDEX', [
        {name: 'names', type: 'INDEX', value: []}
    ]);
    t.names = [];
    for (var i = 0; i < fontNames.length; i += 1) {
        t.names.push({name: 'name_' + i, type: 'NAME', value: fontNames[i]});
    }

    return t;
}

// Given a dictionary's metadata, create a DICT structure.
function makeDict(meta, attrs, strings) {
    var m = {};
    for (var i = 0; i < meta.length; i += 1) {
        var entry = meta[i];
        var value = attrs[entry.name];
        if (value !== undefined && !equals(value, entry.value)) {
            if (entry.type === 'SID') {
                value = encodeString(value, strings);
            }

            m[entry.op] = {name: entry.name, type: entry.type, value: value};
        }
    }

    return m;
}

// The Top DICT houses the global font attributes.
function makeTopDict(attrs, strings) {
    var t = new table.Table('Top DICT', [
        {name: 'dict', type: 'DICT', value: {}}
    ]);
    t.dict = makeDict(TOP_DICT_META, attrs, strings);
    return t;
}

function makeTopDictIndex(topDict) {
    var t = new table.Table('Top DICT INDEX', [
        {name: 'topDicts', type: 'INDEX', value: []}
    ]);
    t.topDicts = [{name: 'topDict_0', type: 'TABLE', value: topDict}];
    return t;
}

function makeStringIndex(strings) {
    var t = new table.Table('String INDEX', [
        {name: 'strings', type: 'INDEX', value: []}
    ]);
    t.strings = [];
    for (var i = 0; i < strings.length; i += 1) {
        t.strings.push({name: 'string_' + i, type: 'STRING', value: strings[i]});
    }

    return t;
}

function makeGlobalSubrIndex() {
    // Currently we don't use subroutines.
    return new table.Table('Global Subr INDEX', [
        {name: 'subrs', type: 'INDEX', value: []}
    ]);
}

function makeCharsets(glyphNames, strings) {
    var t = new table.Table('Charsets', [
        {name: 'format', type: 'Card8', value: 0}
    ]);
    for (var i = 0; i < glyphNames.length; i += 1) {
        var glyphName = glyphNames[i];
        var glyphSID = encodeString(glyphName, strings);
        t.fields.push({name: 'glyph_' + i, type: 'SID', value: glyphSID});
    }

    return t;
}

function glyphToOps(glyph) {
    var ops = [];
    var path = glyph.path;
    ops.push({name: 'width', type: 'NUMBER', value: glyph.advanceWidth});
    var x = 0;
    var y = 0;
    for (var i = 0; i < path.commands.length; i += 1) {
        var dx;
        var dy;
        var cmd = path.commands[i];
        if (cmd.type === 'Q') {
            // CFF only supports bézier curves, so convert the quad to a bézier.
            var _13 = 1 / 3;
            var _23 = 2 / 3;

            // We're going to create a new command so we don't change the original path.
            cmd = {
                type: 'C',
                x: cmd.x,
                y: cmd.y,
                x1: _13 * x + _23 * cmd.x1,
                y1: _13 * y + _23 * cmd.y1,
                x2: _13 * cmd.x + _23 * cmd.x1,
                y2: _13 * cmd.y + _23 * cmd.y1
            };
        }

        if (cmd.type === 'M') {
            dx = Math.round(cmd.x - x);
            dy = Math.round(cmd.y - y);
            ops.push({name: 'dx', type: 'NUMBER', value: dx});
            ops.push({name: 'dy', type: 'NUMBER', value: dy});
            ops.push({name: 'rmoveto', type: 'OP', value: 21});
            x = Math.round(cmd.x);
            y = Math.round(cmd.y);
        } else if (cmd.type === 'L') {
            dx = Math.round(cmd.x - x);
            dy = Math.round(cmd.y - y);
            ops.push({name: 'dx', type: 'NUMBER', value: dx});
            ops.push({name: 'dy', type: 'NUMBER', value: dy});
            ops.push({name: 'rlineto', type: 'OP', value: 5});
            x = Math.round(cmd.x);
            y = Math.round(cmd.y);
        } else if (cmd.type === 'C') {
            var dx1 = Math.round(cmd.x1 - x);
            var dy1 = Math.round(cmd.y1 - y);
            var dx2 = Math.round(cmd.x2 - cmd.x1);
            var dy2 = Math.round(cmd.y2 - cmd.y1);
            dx = Math.round(cmd.x - cmd.x2);
            dy = Math.round(cmd.y - cmd.y2);
            ops.push({name: 'dx1', type: 'NUMBER', value: dx1});
            ops.push({name: 'dy1', type: 'NUMBER', value: dy1});
            ops.push({name: 'dx2', type: 'NUMBER', value: dx2});
            ops.push({name: 'dy2', type: 'NUMBER', value: dy2});
            ops.push({name: 'dx', type: 'NUMBER', value: dx});
            ops.push({name: 'dy', type: 'NUMBER', value: dy});
            ops.push({name: 'rrcurveto', type: 'OP', value: 8});
            x = Math.round(cmd.x);
            y = Math.round(cmd.y);
        }

        // Contours are closed automatically.

    }

    ops.push({name: 'endchar', type: 'OP', value: 14});
    return ops;
}

function makeCharStringsIndex(glyphs) {
    var t = new table.Table('CharStrings INDEX', [
        {name: 'charStrings', type: 'INDEX', value: []}
    ]);

    for (var i = 0; i < glyphs.length; i += 1) {
        var glyph = glyphs.get(i);
        var ops = glyphToOps(glyph);
        t.charStrings.push({name: glyph.name, type: 'CHARSTRING', value: ops});
    }

    return t;
}

function makePrivateDict(attrs, strings) {
    var t = new table.Table('Private DICT', [
        {name: 'dict', type: 'DICT', value: {}}
    ]);
    t.dict = makeDict(PRIVATE_DICT_META, attrs, strings);
    return t;
}

function makePrivateDictIndex(privateDict) {
    var t = new table.Table('Private DICT INDEX', [
        {name: 'privateDicts', type: 'INDEX', value: []}
    ]);
    t.privateDicts = [{name: 'privateDict_0', type: 'TABLE', value: privateDict}];
    return t;
}

function makeCFFTable(glyphs, options) {
    var t = new table.Table('CFF ', [
        {name: 'header', type: 'TABLE'},
        {name: 'nameIndex', type: 'TABLE'},
        {name: 'topDictIndex', type: 'TABLE'},
        {name: 'stringIndex', type: 'TABLE'},
        {name: 'globalSubrIndex', type: 'TABLE'},
        {name: 'charsets', type: 'TABLE'},
        {name: 'charStringsIndex', type: 'TABLE'},
        {name: 'privateDictIndex', type: 'TABLE'}
    ]);

    var fontScale = 1 / options.unitsPerEm;
    // We use non-zero values for the offsets so that the DICT encodes them.
    // This is important because the size of the Top DICT plays a role in offset calculation,
    // and the size shouldn't change after we've written correct offsets.
    var attrs = {
        version: options.version,
        fullName: options.fullName,
        familyName: options.familyName,
        weight: options.weightName,
        fontMatrix: [fontScale, 0, 0, fontScale, 0, 0],
        charset: 999,
        encoding: 0,
        charStrings: 999,
        private: [0, 999]
    };

    var privateAttrs = {};

    var glyphNames = [];
    var glyph;

    // Skip first glyph (.notdef)
    for (var i = 1; i < glyphs.length; i += 1) {
        glyph = glyphs.get(i);
        glyphNames.push(glyph.name);
    }

    var strings = [];

    t.header = makeHeader();
    t.nameIndex = makeNameIndex([options.postScriptName]);
    var topDict = makeTopDict(attrs, strings);
    t.topDictIndex = makeTopDictIndex(topDict);
    t.globalSubrIndex = makeGlobalSubrIndex();
    t.charsets = makeCharsets(glyphNames, strings);
    t.charStringsIndex = makeCharStringsIndex(glyphs);
    var privateDict = makePrivateDict(privateAttrs, strings);
    t.privateDictIndex = makePrivateDictIndex(privateDict);

    // Needs to come at the end, to encode all custom strings used in the font.
    t.stringIndex = makeStringIndex(strings);

    var startOffset = t.header.sizeOf() +
        t.nameIndex.sizeOf() +
        t.topDictIndex.sizeOf() +
        t.stringIndex.sizeOf() +
        t.globalSubrIndex.sizeOf();
    attrs.charset = startOffset;

    // We use the CFF standard encoding; proper encoding will be handled in cmap.
    attrs.encoding = 0;
    attrs.charStrings = attrs.charset + t.charsets.sizeOf();
    attrs.private[1] = attrs.charStrings + t.charStringsIndex.sizeOf();

    // Recreate the Top DICT INDEX with the correct offsets.
    topDict = makeTopDict(attrs, strings);
    t.topDictIndex = makeTopDictIndex(topDict);

    return t;
}

exports.parse = parseCFFTable;
exports.make = makeCFFTable;

},{"../encoding":4,"../glyphset":7,"../parse":9,"../path":10,"../table":11}],13:[function(_dereq_,module,exports){
// The `cmap` table stores the mappings from characters to glyphs.
// https://www.microsoft.com/typography/OTSPEC/cmap.htm

'use strict';

var check = _dereq_('../check');
var parse = _dereq_('../parse');
var table = _dereq_('../table');

// Parse the `cmap` table. This table stores the mappings from characters to glyphs.
// There are many available formats, but we only support the Windows format 4.
// This function returns a `CmapEncoding` object or null if no supported format could be found.
function parseCmapTable(data, start) {
    var i;
    var cmap = {};
    cmap.version = parse.getUShort(data, start);
    check.argument(cmap.version === 0, 'cmap table version should be 0.');

    // The cmap table can contain many sub-tables, each with their own format.
    // We're only interested in a "platform 3" table. This is a Windows format.
    cmap.numTables = parse.getUShort(data, start + 2);
    var offset = -1;
    for (i = 0; i < cmap.numTables; i += 1) {
        var platformId = parse.getUShort(data, start + 4 + (i * 8));
        var encodingId = parse.getUShort(data, start + 4 + (i * 8) + 2);
        if (platformId === 3 && (encodingId === 1 || encodingId === 0)) {
            offset = parse.getULong(data, start + 4 + (i * 8) + 4);
            break;
        }
    }

    if (offset === -1) {
        // There is no cmap table in the font that we support, so return null.
        // This font will be marked as unsupported.
        return null;
    }

    var p = new parse.Parser(data, start + offset);
    cmap.format = p.parseUShort();
    check.argument(cmap.format === 4, 'Only format 4 cmap tables are supported.');

    // Length in bytes of the sub-tables.
    cmap.length = p.parseUShort();
    cmap.language = p.parseUShort();

    // segCount is stored x 2.
    var segCount;
    cmap.segCount = segCount = p.parseUShort() >> 1;

    // Skip searchRange, entrySelector, rangeShift.
    p.skip('uShort', 3);

    // The "unrolled" mapping from character codes to glyph indices.
    cmap.glyphIndexMap = {};

    var endCountParser = new parse.Parser(data, start + offset + 14);
    var startCountParser = new parse.Parser(data, start + offset + 16 + segCount * 2);
    var idDeltaParser = new parse.Parser(data, start + offset + 16 + segCount * 4);
    var idRangeOffsetParser = new parse.Parser(data, start + offset + 16 + segCount * 6);
    var glyphIndexOffset = start + offset + 16 + segCount * 8;
    for (i = 0; i < segCount - 1; i += 1) {
        var glyphIndex;
        var endCount = endCountParser.parseUShort();
        var startCount = startCountParser.parseUShort();
        var idDelta = idDeltaParser.parseShort();
        var idRangeOffset = idRangeOffsetParser.parseUShort();
        for (var c = startCount; c <= endCount; c += 1) {
            if (idRangeOffset !== 0) {
                // The idRangeOffset is relative to the current position in the idRangeOffset array.
                // Take the current offset in the idRangeOffset array.
                glyphIndexOffset = (idRangeOffsetParser.offset + idRangeOffsetParser.relativeOffset - 2);

                // Add the value of the idRangeOffset, which will move us into the glyphIndex array.
                glyphIndexOffset += idRangeOffset;

                // Then add the character index of the current segment, multiplied by 2 for USHORTs.
                glyphIndexOffset += (c - startCount) * 2;
                glyphIndex = parse.getUShort(data, glyphIndexOffset);
                if (glyphIndex !== 0) {
                    glyphIndex = (glyphIndex + idDelta) & 0xFFFF;
                }
            } else {
                glyphIndex = (c + idDelta) & 0xFFFF;
            }

            cmap.glyphIndexMap[c] = glyphIndex;
        }
    }

    return cmap;
}

function addSegment(t, code, glyphIndex) {
    t.segments.push({
        end: code,
        start: code,
        delta: -(code - glyphIndex),
        offset: 0
    });
}

function addTerminatorSegment(t) {
    t.segments.push({
        end: 0xFFFF,
        start: 0xFFFF,
        delta: 1,
        offset: 0
    });
}

function makeCmapTable(glyphs) {
    var i;
    var t = new table.Table('cmap', [
        {name: 'version', type: 'USHORT', value: 0},
        {name: 'numTables', type: 'USHORT', value: 1},
        {name: 'platformID', type: 'USHORT', value: 3},
        {name: 'encodingID', type: 'USHORT', value: 1},
        {name: 'offset', type: 'ULONG', value: 12},
        {name: 'format', type: 'USHORT', value: 4},
        {name: 'length', type: 'USHORT', value: 0},
        {name: 'language', type: 'USHORT', value: 0},
        {name: 'segCountX2', type: 'USHORT', value: 0},
        {name: 'searchRange', type: 'USHORT', value: 0},
        {name: 'entrySelector', type: 'USHORT', value: 0},
        {name: 'rangeShift', type: 'USHORT', value: 0}
    ]);

    t.segments = [];
    for (i = 0; i < glyphs.length; i += 1) {
        var glyph = glyphs.get(i);
        for (var j = 0; j < glyph.unicodes.length; j += 1) {
            addSegment(t, glyph.unicodes[j], i);
        }

        t.segments = t.segments.sort(function(a, b) {
            return a.start - b.start;
        });
    }

    addTerminatorSegment(t);

    var segCount;
    segCount = t.segments.length;
    t.segCountX2 = segCount * 2;
    t.searchRange = Math.pow(2, Math.floor(Math.log(segCount) / Math.log(2))) * 2;
    t.entrySelector = Math.log(t.searchRange / 2) / Math.log(2);
    t.rangeShift = t.segCountX2 - t.searchRange;

    // Set up parallel segment arrays.
    var endCounts = [];
    var startCounts = [];
    var idDeltas = [];
    var idRangeOffsets = [];
    var glyphIds = [];

    for (i = 0; i < segCount; i += 1) {
        var segment = t.segments[i];
        endCounts = endCounts.concat({name: 'end_' + i, type: 'USHORT', value: segment.end});
        startCounts = startCounts.concat({name: 'start_' + i, type: 'USHORT', value: segment.start});
        idDeltas = idDeltas.concat({name: 'idDelta_' + i, type: 'SHORT', value: segment.delta});
        idRangeOffsets = idRangeOffsets.concat({name: 'idRangeOffset_' + i, type: 'USHORT', value: segment.offset});
        if (segment.glyphId !== undefined) {
            glyphIds = glyphIds.concat({name: 'glyph_' + i, type: 'USHORT', value: segment.glyphId});
        }
    }

    t.fields = t.fields.concat(endCounts);
    t.fields.push({name: 'reservedPad', type: 'USHORT', value: 0});
    t.fields = t.fields.concat(startCounts);
    t.fields = t.fields.concat(idDeltas);
    t.fields = t.fields.concat(idRangeOffsets);
    t.fields = t.fields.concat(glyphIds);

    t.length = 14 + // Subtable header
        endCounts.length * 2 +
        2 + // reservedPad
        startCounts.length * 2 +
        idDeltas.length * 2 +
        idRangeOffsets.length * 2 +
        glyphIds.length * 2;

    return t;
}

exports.parse = parseCmapTable;
exports.make = makeCmapTable;

},{"../check":2,"../parse":9,"../table":11}],14:[function(_dereq_,module,exports){
// The `glyf` table describes the glyphs in TrueType outline format.
// http://www.microsoft.com/typography/otspec/glyf.htm

'use strict';

var check = _dereq_('../check');
var glyphset = _dereq_('../glyphset');
var parse = _dereq_('../parse');
var path = _dereq_('../path');

// Parse the coordinate data for a glyph.
function parseGlyphCoordinate(p, flag, previousValue, shortVectorBitMask, sameBitMask) {
    var v;
    if ((flag & shortVectorBitMask) > 0) {
        // The coordinate is 1 byte long.
        v = p.parseByte();
        // The `same` bit is re-used for short values to signify the sign of the value.
        if ((flag & sameBitMask) === 0) {
            v = -v;
        }

        v = previousValue + v;
    } else {
        //  The coordinate is 2 bytes long.
        // If the `same` bit is set, the coordinate is the same as the previous coordinate.
        if ((flag & sameBitMask) > 0) {
            v = previousValue;
        } else {
            // Parse the coordinate as a signed 16-bit delta value.
            v = previousValue + p.parseShort();
        }
    }

    return v;
}

// Parse a TrueType glyph.
function parseGlyph(glyph, data, start) {
    var p = new parse.Parser(data, start);
    glyph.numberOfContours = p.parseShort();
    glyph.xMin = p.parseShort();
    glyph.yMin = p.parseShort();
    glyph.xMax = p.parseShort();
    glyph.yMax = p.parseShort();
    var flags;
    var flag;
    if (glyph.numberOfContours > 0) {
        var i;
        // This glyph is not a composite.
        var endPointIndices = glyph.endPointIndices = [];
        for (i = 0; i < glyph.numberOfContours; i += 1) {
            endPointIndices.push(p.parseUShort());
        }

        glyph.instructionLength = p.parseUShort();
        glyph.instructions = [];
        for (i = 0; i < glyph.instructionLength; i += 1) {
            glyph.instructions.push(p.parseByte());
        }

        var numberOfCoordinates = endPointIndices[endPointIndices.length - 1] + 1;
        flags = [];
        for (i = 0; i < numberOfCoordinates; i += 1) {
            flag = p.parseByte();
            flags.push(flag);
            // If bit 3 is set, we repeat this flag n times, where n is the next byte.
            if ((flag & 8) > 0) {
                var repeatCount = p.parseByte();
                for (var j = 0; j < repeatCount; j += 1) {
                    flags.push(flag);
                    i += 1;
                }
            }
        }

        check.argument(flags.length === numberOfCoordinates, 'Bad flags.');

        if (endPointIndices.length > 0) {
            var points = [];
            var point;
            // X/Y coordinates are relative to the previous point, except for the first point which is relative to 0,0.
            if (numberOfCoordinates > 0) {
                for (i = 0; i < numberOfCoordinates; i += 1) {
                    flag = flags[i];
                    point = {};
                    point.onCurve = !!(flag & 1);
                    point.lastPointOfContour = endPointIndices.indexOf(i) >= 0;
                    points.push(point);
                }

                var px = 0;
                for (i = 0; i < numberOfCoordinates; i += 1) {
                    flag = flags[i];
                    point = points[i];
                    point.x = parseGlyphCoordinate(p, flag, px, 2, 16);
                    px = point.x;
                }

                var py = 0;
                for (i = 0; i < numberOfCoordinates; i += 1) {
                    flag = flags[i];
                    point = points[i];
                    point.y = parseGlyphCoordinate(p, flag, py, 4, 32);
                    py = point.y;
                }
            }

            glyph.points = points;
        } else {
            glyph.points = [];
        }
    } else if (glyph.numberOfContours === 0) {
        glyph.points = [];
    } else {
        glyph.isComposite = true;
        glyph.points = [];
        glyph.components = [];
        var moreComponents = true;
        while (moreComponents) {
            flags = p.parseUShort();
            var component = {
                glyphIndex: p.parseUShort(),
                xScale: 1,
                scale01: 0,
                scale10: 0,
                yScale: 1,
                dx: 0,
                dy: 0
            };
            if ((flags & 1) > 0) {
                // The arguments are words
                component.dx = p.parseShort();
                component.dy = p.parseShort();
            } else {
                // The arguments are bytes
                component.dx = p.parseChar();
                component.dy = p.parseChar();
            }

            if ((flags & 8) > 0) {
                // We have a scale
                component.xScale = component.yScale = p.parseF2Dot14();
            } else if ((flags & 64) > 0) {
                // We have an X / Y scale
                component.xScale = p.parseF2Dot14();
                component.yScale = p.parseF2Dot14();
            } else if ((flags & 128) > 0) {
                // We have a 2x2 transformation
                component.xScale = p.parseF2Dot14();
                component.scale01 = p.parseF2Dot14();
                component.scale10 = p.parseF2Dot14();
                component.yScale = p.parseF2Dot14();
            }

            glyph.components.push(component);
            moreComponents = !!(flags & 32);
        }
    }
}

// Transform an array of points and return a new array.
function transformPoints(points, transform) {
    var newPoints = [];
    for (var i = 0; i < points.length; i += 1) {
        var pt = points[i];
        var newPt = {
            x: transform.xScale * pt.x + transform.scale01 * pt.y + transform.dx,
            y: transform.scale10 * pt.x + transform.yScale * pt.y + transform.dy,
            onCurve: pt.onCurve,
            lastPointOfContour: pt.lastPointOfContour
        };
        newPoints.push(newPt);
    }

    return newPoints;
}

function getContours(points) {
    var contours = [];
    var currentContour = [];
    for (var i = 0; i < points.length; i += 1) {
        var pt = points[i];
        currentContour.push(pt);
        if (pt.lastPointOfContour) {
            contours.push(currentContour);
            currentContour = [];
        }
    }

    check.argument(currentContour.length === 0, 'There are still points left in the current contour.');
    return contours;
}

// Convert the TrueType glyph outline to a Path.
function getPath(points) {
    var p = new path.Path();
    if (!points) {
        return p;
    }

    var contours = getContours(points);
    for (var i = 0; i < contours.length; i += 1) {
        var contour = contours[i];
        var firstPt = contour[0];
        var lastPt = contour[contour.length - 1];
        var curvePt;
        var realFirstPoint;
        if (firstPt.onCurve) {
            curvePt = null;
            // The first point will be consumed by the moveTo command,
            // so skip it in the loop.
            realFirstPoint = true;
        } else {
            if (lastPt.onCurve) {
                // If the first point is off-curve and the last point is on-curve,
                // start at the last point.
                firstPt = lastPt;
            } else {
                // If both first and last points are off-curve, start at their middle.
                firstPt = { x: (firstPt.x + lastPt.x) / 2, y: (firstPt.y + lastPt.y) / 2 };
            }

            curvePt = firstPt;
            // The first point is synthesized, so don't skip the real first point.
            realFirstPoint = false;
        }

        p.moveTo(firstPt.x, firstPt.y);

        for (var j = realFirstPoint ? 1 : 0; j < contour.length; j += 1) {
            var pt = contour[j];
            var prevPt = j === 0 ? firstPt : contour[j - 1];
            if (prevPt.onCurve && pt.onCurve) {
                // This is a straight line.
                p.lineTo(pt.x, pt.y);
            } else if (prevPt.onCurve && !pt.onCurve) {
                curvePt = pt;
            } else if (!prevPt.onCurve && !pt.onCurve) {
                var midPt = { x: (prevPt.x + pt.x) / 2, y: (prevPt.y + pt.y) / 2 };
                p.quadraticCurveTo(prevPt.x, prevPt.y, midPt.x, midPt.y);
                curvePt = pt;
            } else if (!prevPt.onCurve && pt.onCurve) {
                // Previous point off-curve, this point on-curve.
                p.quadraticCurveTo(curvePt.x, curvePt.y, pt.x, pt.y);
                curvePt = null;
            } else {
                throw new Error('Invalid state.');
            }
        }

        if (firstPt !== lastPt) {
            // Connect the last and first points
            if (curvePt) {
                p.quadraticCurveTo(curvePt.x, curvePt.y, firstPt.x, firstPt.y);
            } else {
                p.lineTo(firstPt.x, firstPt.y);
            }
        }
    }

    p.closePath();
    return p;
}

function buildPath(glyphs, glyph) {
    if (glyph.isComposite) {
        for (var j = 0; j < glyph.components.length; j += 1) {
            var component = glyph.components[j];
            var componentGlyph = glyphs.get(component.glyphIndex);
            if (componentGlyph.points) {
                var transformedPoints = transformPoints(componentGlyph.points, component);
                glyph.points = glyph.points.concat(transformedPoints);
            }
        }
    }

    return getPath(glyph.points);
}

// Parse all the glyphs according to the offsets from the `loca` table.
function parseGlyfTable(data, start, loca, font) {
    var glyphs = new glyphset.GlyphSet(font);
    var i;

    // The last element of the loca table is invalid.
    for (i = 0; i < loca.length - 1; i += 1) {
        var offset = loca[i];
        var nextOffset = loca[i + 1];
        if (offset !== nextOffset) {
            glyphs.push(i, glyphset.ttfGlyphLoader(font, i, parseGlyph, data, start + offset, buildPath));
        } else {
            glyphs.push(i, glyphset.glyphLoader(font, i));
        }
    }

    return glyphs;
}

exports.parse = parseGlyfTable;

},{"../check":2,"../glyphset":7,"../parse":9,"../path":10}],15:[function(_dereq_,module,exports){
// The `GPOS` table contains kerning pairs, among other things.
// https://www.microsoft.com/typography/OTSPEC/gpos.htm

'use strict';

var check = _dereq_('../check');
var parse = _dereq_('../parse');

// Parse ScriptList and FeatureList tables of GPOS, GSUB, GDEF, BASE, JSTF tables.
// These lists are unused by now, this function is just the basis for a real parsing.
function parseTaggedListTable(data, start) {
    var p = new parse.Parser(data, start);
    var n = p.parseUShort();
    var list = [];
    for (var i = 0; i < n; i++) {
        list[p.parseTag()] = { offset: p.parseUShort() };
    }

    return list;
}

// Parse a coverage table in a GSUB, GPOS or GDEF table.
// Format 1 is a simple list of glyph ids,
// Format 2 is a list of ranges. It is expanded in a list of glyphs, maybe not the best idea.
function parseCoverageTable(data, start) {
    var p = new parse.Parser(data, start);
    var format = p.parseUShort();
    var count =  p.parseUShort();
    if (format === 1) {
        return p.parseUShortList(count);
    }
    else if (format === 2) {
        var coverage = [];
        for (; count--;) {
            var begin = p.parseUShort();
            var end = p.parseUShort();
            var index = p.parseUShort();
            for (var i = begin; i <= end; i++) {
                coverage[index++] = i;
            }
        }

        return coverage;
    }
}

// Parse a Class Definition Table in a GSUB, GPOS or GDEF table.
// Returns a function that gets a class value from a glyph ID.
function parseClassDefTable(data, start) {
    var p = new parse.Parser(data, start);
    var format = p.parseUShort();
    if (format === 1) {
        // Format 1 specifies a range of consecutive glyph indices, one class per glyph ID.
        var startGlyph = p.parseUShort();
        var glyphCount = p.parseUShort();
        var classes = p.parseUShortList(glyphCount);
        return function(glyphID) {
            return classes[glyphID - startGlyph] || 0;
        };
    }
    else if (format === 2) {
        // Format 2 defines multiple groups of glyph indices that belong to the same class.
        var rangeCount = p.parseUShort();
        var startGlyphs = [];
        var endGlyphs = [];
        var classValues = [];
        for (var i = 0; i < rangeCount; i++) {
            startGlyphs[i] = p.parseUShort();
            endGlyphs[i] = p.parseUShort();
            classValues[i] = p.parseUShort();
        }

        return function(glyphID) {
            var l = 0;
            var r = startGlyphs.length - 1;
            while (l < r) {
                var c = (l + r + 1) >> 1;
                if (glyphID < startGlyphs[c]) {
                    r = c - 1;
                } else {
                    l = c;
                }
            }

            if (startGlyphs[l] <= glyphID && glyphID <= endGlyphs[l]) {
                return classValues[l] || 0;
            }

            return 0;
        };
    }
}

// Parse a pair adjustment positioning subtable, format 1 or format 2
// The subtable is returned in the form of a lookup function.
function parsePairPosSubTable(data, start) {
    var p = new parse.Parser(data, start);
    // This part is common to format 1 and format 2 subtables
    var format = p.parseUShort();
    var coverageOffset = p.parseUShort();
    var coverage = parseCoverageTable(data, start + coverageOffset);
    // valueFormat 4: XAdvance only, 1: XPlacement only, 0: no ValueRecord for second glyph
    // Only valueFormat1=4 and valueFormat2=0 is supported.
    var valueFormat1 = p.parseUShort();
    var valueFormat2 = p.parseUShort();
    var value1;
    var value2;
    if (valueFormat1 !== 4 || valueFormat2 !== 0) return;
    var sharedPairSets = {};
    if (format === 1) {
        // Pair Positioning Adjustment: Format 1
        var pairSetCount = p.parseUShort();
        var pairSet = [];
        // Array of offsets to PairSet tables-from beginning of PairPos subtable-ordered by Coverage Index
        var pairSetOffsets = p.parseOffset16List(pairSetCount);
        for (var firstGlyph = 0; firstGlyph < pairSetCount; firstGlyph++) {
            var pairSetOffset = pairSetOffsets[firstGlyph];
            var sharedPairSet = sharedPairSets[pairSetOffset];
            if (!sharedPairSet) {
                // Parse a pairset table in a pair adjustment subtable format 1
                sharedPairSet = {};
                p.relativeOffset = pairSetOffset;
                var pairValueCount = p.parseUShort();
                for (; pairValueCount--;) {
                    var secondGlyph = p.parseUShort();
                    if (valueFormat1) value1 = p.parseShort();
                    if (valueFormat2) value2 = p.parseShort();
                    // We only support valueFormat1 = 4 and valueFormat2 = 0,
                    // so value1 is the XAdvance and value2 is empty.
                    sharedPairSet[secondGlyph] = value1;
                }
            }

            pairSet[coverage[firstGlyph]] = sharedPairSet;
        }

        return function(leftGlyph, rightGlyph) {
            var pairs = pairSet[leftGlyph];
            if (pairs) return pairs[rightGlyph];
        };
    }
    else if (format === 2) {
        // Pair Positioning Adjustment: Format 2
        var classDef1Offset = p.parseUShort();
        var classDef2Offset = p.parseUShort();
        var class1Count = p.parseUShort();
        var class2Count = p.parseUShort();
        var getClass1 = parseClassDefTable(data, start + classDef1Offset);
        var getClass2 = parseClassDefTable(data, start + classDef2Offset);

        // Parse kerning values by class pair.
        var kerningMatrix = [];
        for (var i = 0; i < class1Count; i++) {
            var kerningRow = kerningMatrix[i] = [];
            for (var j = 0; j < class2Count; j++) {
                if (valueFormat1) value1 = p.parseShort();
                if (valueFormat2) value2 = p.parseShort();
                // We only support valueFormat1 = 4 and valueFormat2 = 0,
                // so value1 is the XAdvance and value2 is empty.
                kerningRow[j] = value1;
            }
        }

        // Convert coverage list to a hash
        var covered = {};
        for (i = 0; i < coverage.length; i++) covered[coverage[i]] = 1;

        // Get the kerning value for a specific glyph pair.
        return function(leftGlyph, rightGlyph) {
            if (!covered[leftGlyph]) return;
            var class1 = getClass1(leftGlyph);
            var class2 = getClass2(rightGlyph);
            var kerningRow = kerningMatrix[class1];

            if (kerningRow) {
                return kerningRow[class2];
            }
        };
    }
}

// Parse a LookupTable (present in of GPOS, GSUB, GDEF, BASE, JSTF tables).
function parseLookupTable(data, start) {
    var p = new parse.Parser(data, start);
    var lookupType = p.parseUShort();
    var lookupFlag = p.parseUShort();
    var useMarkFilteringSet = lookupFlag & 0x10;
    var subTableCount = p.parseUShort();
    var subTableOffsets = p.parseOffset16List(subTableCount);
    var table = {
        lookupType: lookupType,
        lookupFlag: lookupFlag,
        markFilteringSet: useMarkFilteringSet ? p.parseUShort() : -1
    };
    // LookupType 2, Pair adjustment
    if (lookupType === 2) {
        var subtables = [];
        for (var i = 0; i < subTableCount; i++) {
            subtables.push(parsePairPosSubTable(data, start + subTableOffsets[i]));
        }
        // Return a function which finds the kerning values in the subtables.
        table.getKerningValue = function(leftGlyph, rightGlyph) {
            for (var i = subtables.length; i--;) {
                var value = subtables[i](leftGlyph, rightGlyph);
                if (value !== undefined) return value;
            }

            return 0;
        };
    }

    return table;
}

// Parse the `GPOS` table which contains, among other things, kerning pairs.
// https://www.microsoft.com/typography/OTSPEC/gpos.htm
function parseGposTable(data, start, font) {
    var p = new parse.Parser(data, start);
    var tableVersion = p.parseFixed();
    check.argument(tableVersion === 1, 'Unsupported GPOS table version.');

    // ScriptList and FeatureList - ignored for now
    parseTaggedListTable(data, start + p.parseUShort());
    // 'kern' is the feature we are looking for.
    parseTaggedListTable(data, start + p.parseUShort());

    // LookupList
    var lookupListOffset = p.parseUShort();
    p.relativeOffset = lookupListOffset;
    var lookupCount = p.parseUShort();
    var lookupTableOffsets = p.parseOffset16List(lookupCount);
    var lookupListAbsoluteOffset = start + lookupListOffset;
    for (var i = 0; i < lookupCount; i++) {
        var table = parseLookupTable(data, lookupListAbsoluteOffset + lookupTableOffsets[i]);
        if (table.lookupType === 2 && !font.getGposKerningValue) font.getGposKerningValue = table.getKerningValue;
    }
}

exports.parse = parseGposTable;

},{"../check":2,"../parse":9}],16:[function(_dereq_,module,exports){
// The `head` table contains global information about the font.
// https://www.microsoft.com/typography/OTSPEC/head.htm

'use strict';

var check = _dereq_('../check');
var parse = _dereq_('../parse');
var table = _dereq_('../table');

// Parse the header `head` table
function parseHeadTable(data, start) {
    var head = {};
    var p = new parse.Parser(data, start);
    head.version = p.parseVersion();
    head.fontRevision = Math.round(p.parseFixed() * 1000) / 1000;
    head.checkSumAdjustment = p.parseULong();
    head.magicNumber = p.parseULong();
    check.argument(head.magicNumber === 0x5F0F3CF5, 'Font header has wrong magic number.');
    head.flags = p.parseUShort();
    head.unitsPerEm = p.parseUShort();
    head.created = p.parseLongDateTime();
    head.modified = p.parseLongDateTime();
    head.xMin = p.parseShort();
    head.yMin = p.parseShort();
    head.xMax = p.parseShort();
    head.yMax = p.parseShort();
    head.macStyle = p.parseUShort();
    head.lowestRecPPEM = p.parseUShort();
    head.fontDirectionHint = p.parseShort();
    head.indexToLocFormat = p.parseShort();     // 50
    head.glyphDataFormat = p.parseShort();
    return head;
}

function makeHeadTable(options) {
    return new table.Table('head', [
        {name: 'version', type: 'FIXED', value: 0x00010000},
        {name: 'fontRevision', type: 'FIXED', value: 0x00010000},
        {name: 'checkSumAdjustment', type: 'ULONG', value: 0},
        {name: 'magicNumber', type: 'ULONG', value: 0x5F0F3CF5},
        {name: 'flags', type: 'USHORT', value: 0},
        {name: 'unitsPerEm', type: 'USHORT', value: 1000},
        {name: 'created', type: 'LONGDATETIME', value: 0},
        {name: 'modified', type: 'LONGDATETIME', value: 0},
        {name: 'xMin', type: 'SHORT', value: 0},
        {name: 'yMin', type: 'SHORT', value: 0},
        {name: 'xMax', type: 'SHORT', value: 0},
        {name: 'yMax', type: 'SHORT', value: 0},
        {name: 'macStyle', type: 'USHORT', value: 0},
        {name: 'lowestRecPPEM', type: 'USHORT', value: 0},
        {name: 'fontDirectionHint', type: 'SHORT', value: 2},
        {name: 'indexToLocFormat', type: 'SHORT', value: 0},
        {name: 'glyphDataFormat', type: 'SHORT', value: 0}
    ], options);
}

exports.parse = parseHeadTable;
exports.make = makeHeadTable;

},{"../check":2,"../parse":9,"../table":11}],17:[function(_dereq_,module,exports){
// The `hhea` table contains information for horizontal layout.
// https://www.microsoft.com/typography/OTSPEC/hhea.htm

'use strict';

var parse = _dereq_('../parse');
var table = _dereq_('../table');

// Parse the horizontal header `hhea` table
function parseHheaTable(data, start) {
    var hhea = {};
    var p = new parse.Parser(data, start);
    hhea.version = p.parseVersion();
    hhea.ascender = p.parseShort();
    hhea.descender = p.parseShort();
    hhea.lineGap = p.parseShort();
    hhea.advanceWidthMax = p.parseUShort();
    hhea.minLeftSideBearing = p.parseShort();
    hhea.minRightSideBearing = p.parseShort();
    hhea.xMaxExtent = p.parseShort();
    hhea.caretSlopeRise = p.parseShort();
    hhea.caretSlopeRun = p.parseShort();
    hhea.caretOffset = p.parseShort();
    p.relativeOffset += 8;
    hhea.metricDataFormat = p.parseShort();
    hhea.numberOfHMetrics = p.parseUShort();
    return hhea;
}

function makeHheaTable(options) {
    return new table.Table('hhea', [
        {name: 'version', type: 'FIXED', value: 0x00010000},
        {name: 'ascender', type: 'FWORD', value: 0},
        {name: 'descender', type: 'FWORD', value: 0},
        {name: 'lineGap', type: 'FWORD', value: 0},
        {name: 'advanceWidthMax', type: 'UFWORD', value: 0},
        {name: 'minLeftSideBearing', type: 'FWORD', value: 0},
        {name: 'minRightSideBearing', type: 'FWORD', value: 0},
        {name: 'xMaxExtent', type: 'FWORD', value: 0},
        {name: 'caretSlopeRise', type: 'SHORT', value: 1},
        {name: 'caretSlopeRun', type: 'SHORT', value: 0},
        {name: 'caretOffset', type: 'SHORT', value: 0},
        {name: 'reserved1', type: 'SHORT', value: 0},
        {name: 'reserved2', type: 'SHORT', value: 0},
        {name: 'reserved3', type: 'SHORT', value: 0},
        {name: 'reserved4', type: 'SHORT', value: 0},
        {name: 'metricDataFormat', type: 'SHORT', value: 0},
        {name: 'numberOfHMetrics', type: 'USHORT', value: 0}
    ], options);
}

exports.parse = parseHheaTable;
exports.make = makeHheaTable;

},{"../parse":9,"../table":11}],18:[function(_dereq_,module,exports){
// The `hmtx` table contains the horizontal metrics for all glyphs.
// https://www.microsoft.com/typography/OTSPEC/hmtx.htm

'use strict';

var parse = _dereq_('../parse');
var table = _dereq_('../table');

// Parse the `hmtx` table, which contains the horizontal metrics for all glyphs.
// This function augments the glyph array, adding the advanceWidth and leftSideBearing to each glyph.
function parseHmtxTable(data, start, numMetrics, numGlyphs, glyphs) {
    var advanceWidth;
    var leftSideBearing;
    var p = new parse.Parser(data, start);
    for (var i = 0; i < numGlyphs; i += 1) {
        // If the font is monospaced, only one entry is needed. This last entry applies to all subsequent glyphs.
        if (i < numMetrics) {
            advanceWidth = p.parseUShort();
            leftSideBearing = p.parseShort();
        }

        var glyph = glyphs.get(i);
        glyph.advanceWidth = advanceWidth;
        glyph.leftSideBearing = leftSideBearing;
    }
}

function makeHmtxTable(glyphs) {
    var t = new table.Table('hmtx', []);
    for (var i = 0; i < glyphs.length; i += 1) {
        var glyph = glyphs.get(i);
        var advanceWidth = glyph.advanceWidth || 0;
        var leftSideBearing = glyph.leftSideBearing || 0;
        t.fields.push({name: 'advanceWidth_' + i, type: 'USHORT', value: advanceWidth});
        t.fields.push({name: 'leftSideBearing_' + i, type: 'SHORT', value: leftSideBearing});
    }

    return t;
}

exports.parse = parseHmtxTable;
exports.make = makeHmtxTable;

},{"../parse":9,"../table":11}],19:[function(_dereq_,module,exports){
// The `kern` table contains kerning pairs.
// Note that some fonts use the GPOS OpenType layout table to specify kerning.
// https://www.microsoft.com/typography/OTSPEC/kern.htm

'use strict';

var check = _dereq_('../check');
var parse = _dereq_('../parse');

// Parse the `kern` table which contains kerning pairs.
function parseKernTable(data, start) {
    var pairs = {};
    var p = new parse.Parser(data, start);
    var tableVersion = p.parseUShort();
    check.argument(tableVersion === 0, 'Unsupported kern table version.');
    // Skip nTables.
    p.skip('uShort', 1);
    var subTableVersion = p.parseUShort();
    check.argument(subTableVersion === 0, 'Unsupported kern sub-table version.');
    // Skip subTableLength, subTableCoverage
    p.skip('uShort', 2);
    var nPairs = p.parseUShort();
    // Skip searchRange, entrySelector, rangeShift.
    p.skip('uShort', 3);
    for (var i = 0; i < nPairs; i += 1) {
        var leftIndex = p.parseUShort();
        var rightIndex = p.parseUShort();
        var value = p.parseShort();
        pairs[leftIndex + ',' + rightIndex] = value;
    }

    return pairs;
}

exports.parse = parseKernTable;

},{"../check":2,"../parse":9}],20:[function(_dereq_,module,exports){
// The `loca` table stores the offsets to the locations of the glyphs in the font.
// https://www.microsoft.com/typography/OTSPEC/loca.htm

'use strict';

var parse = _dereq_('../parse');

// Parse the `loca` table. This table stores the offsets to the locations of the glyphs in the font,
// relative to the beginning of the glyphData table.
// The number of glyphs stored in the `loca` table is specified in the `maxp` table (under numGlyphs)
// The loca table has two versions: a short version where offsets are stored as uShorts, and a long
// version where offsets are stored as uLongs. The `head` table specifies which version to use
// (under indexToLocFormat).
function parseLocaTable(data, start, numGlyphs, shortVersion) {
    var p = new parse.Parser(data, start);
    var parseFn = shortVersion ? p.parseUShort : p.parseULong;
    // There is an extra entry after the last index element to compute the length of the last glyph.
    // That's why we use numGlyphs + 1.
    var glyphOffsets = [];
    for (var i = 0; i < numGlyphs + 1; i += 1) {
        var glyphOffset = parseFn.call(p);
        if (shortVersion) {
            // The short table version stores the actual offset divided by 2.
            glyphOffset *= 2;
        }

        glyphOffsets.push(glyphOffset);
    }

    return glyphOffsets;
}

exports.parse = parseLocaTable;

},{"../parse":9}],21:[function(_dereq_,module,exports){
// The `maxp` table establishes the memory requirements for the font.
// We need it just to get the number of glyphs in the font.
// https://www.microsoft.com/typography/OTSPEC/maxp.htm

'use strict';

var parse = _dereq_('../parse');
var table = _dereq_('../table');

// Parse the maximum profile `maxp` table.
function parseMaxpTable(data, start) {
    var maxp = {};
    var p = new parse.Parser(data, start);
    maxp.version = p.parseVersion();
    maxp.numGlyphs = p.parseUShort();
    if (maxp.version === 1.0) {
        maxp.maxPoints = p.parseUShort();
        maxp.maxContours = p.parseUShort();
        maxp.maxCompositePoints = p.parseUShort();
        maxp.maxCompositeContours = p.parseUShort();
        maxp.maxZones = p.parseUShort();
        maxp.maxTwilightPoints = p.parseUShort();
        maxp.maxStorage = p.parseUShort();
        maxp.maxFunctionDefs = p.parseUShort();
        maxp.maxInstructionDefs = p.parseUShort();
        maxp.maxStackElements = p.parseUShort();
        maxp.maxSizeOfInstructions = p.parseUShort();
        maxp.maxComponentElements = p.parseUShort();
        maxp.maxComponentDepth = p.parseUShort();
    }

    return maxp;
}

function makeMaxpTable(numGlyphs) {
    return new table.Table('maxp', [
        {name: 'version', type: 'FIXED', value: 0x00005000},
        {name: 'numGlyphs', type: 'USHORT', value: numGlyphs}
    ]);
}

exports.parse = parseMaxpTable;
exports.make = makeMaxpTable;

},{"../parse":9,"../table":11}],22:[function(_dereq_,module,exports){
// The `name` naming table.
// https://www.microsoft.com/typography/OTSPEC/name.htm

'use strict';

var encode = _dereq_('../types').encode;
var parse = _dereq_('../parse');
var table = _dereq_('../table');

// NameIDs for the name table.
var nameTableNames = [
    'copyright',              // 0
    'fontFamily',             // 1
    'fontSubfamily',          // 2
    'uniqueID',               // 3
    'fullName',               // 4
    'version',                // 5
    'postScriptName',         // 6
    'trademark',              // 7
    'manufacturer',           // 8
    'designer',               // 9
    'description',            // 10
    'manufacturerURL',        // 11
    'designerURL',            // 12
    'licence',                // 13
    'licenceURL',             // 14
    'reserved',               // 15
    'preferredFamily',        // 16
    'preferredSubfamily',     // 17
    'compatibleFullName',     // 18
    'sampleText',             // 19
    'postScriptFindFontName', // 20
    'wwsFamily',              // 21
    'wwsSubfamily'            // 22
];

// Parse the naming `name` table
// Only Windows Unicode English names are supported.
// Format 1 additional fields are not supported
function parseNameTable(data, start) {
    var name = {};
    var p = new parse.Parser(data, start);
    name.format = p.parseUShort();
    var count = p.parseUShort();
    var stringOffset = p.offset + p.parseUShort();
    var unknownCount = 0;
    for (var i = 0; i < count; i++) {
        var platformID = p.parseUShort();
        var encodingID = p.parseUShort();
        var languageID = p.parseUShort();
        var nameID = p.parseUShort();
        var property = nameTableNames[nameID];
        var byteLength = p.parseUShort();
        var offset = p.parseUShort();
        // platformID - encodingID - languageID standard combinations :
        // 1 - 0 - 0 : Macintosh, Roman, English
        // 3 - 1 - 0x409 : Windows, Unicode BMP (UCS-2), en-US
        if (platformID === 3 && encodingID === 1 && languageID === 0x409) {
            var codePoints = [];
            var length = byteLength / 2;
            for (var j = 0; j < length; j++, offset += 2) {
                codePoints[j] = parse.getShort(data, stringOffset + offset);
            }

            var str = String.fromCharCode.apply(null, codePoints);
            if (property) {
                name[property] = str;
            }
            else {
                unknownCount++;
                name['unknown' + unknownCount] = str;
            }
        }

    }

    if (name.format === 1) {
        name.langTagCount = p.parseUShort();
    }

    return name;
}

function makeNameRecord(platformID, encodingID, languageID, nameID, length, offset) {
    return new table.Table('NameRecord', [
        {name: 'platformID', type: 'USHORT', value: platformID},
        {name: 'encodingID', type: 'USHORT', value: encodingID},
        {name: 'languageID', type: 'USHORT', value: languageID},
        {name: 'nameID', type: 'USHORT', value: nameID},
        {name: 'length', type: 'USHORT', value: length},
        {name: 'offset', type: 'USHORT', value: offset}
    ]);
}

function addMacintoshNameRecord(t, recordID, s, offset) {
    // Macintosh, Roman, English
    var stringBytes = encode.STRING(s);
    t.records.push(makeNameRecord(1, 0, 0, recordID, stringBytes.length, offset));
    t.strings.push(stringBytes);
    offset += stringBytes.length;
    return offset;
}

function addWindowsNameRecord(t, recordID, s, offset) {
    // Windows, Unicode BMP (UCS-2), US English
    var utf16Bytes = encode.UTF16(s);
    t.records.push(makeNameRecord(3, 1, 0x0409, recordID, utf16Bytes.length, offset));
    t.strings.push(utf16Bytes);
    offset += utf16Bytes.length;
    return offset;
}

function makeNameTable(options) {
    var t = new table.Table('name', [
        {name: 'format', type: 'USHORT', value: 0},
        {name: 'count', type: 'USHORT', value: 0},
        {name: 'stringOffset', type: 'USHORT', value: 0}
    ]);
    t.records = [];
    t.strings = [];
    var offset = 0;
    var i;
    var s;
    // Add Macintosh records first
    for (i = 0; i < nameTableNames.length; i += 1) {
        if (options[nameTableNames[i]] !== undefined) {
            s = options[nameTableNames[i]];
            offset = addMacintoshNameRecord(t, i, s, offset);
        }
    }
    // Then add Windows records
    for (i = 0; i < nameTableNames.length; i += 1) {
        if (options[nameTableNames[i]] !== undefined) {
            s = options[nameTableNames[i]];
            offset = addWindowsNameRecord(t, i, s, offset);
        }
    }

    t.count = t.records.length;
    t.stringOffset = 6 + t.count * 12;
    for (i = 0; i < t.records.length; i += 1) {
        t.fields.push({name: 'record_' + i, type: 'TABLE', value: t.records[i]});
    }

    for (i = 0; i < t.strings.length; i += 1) {
        t.fields.push({name: 'string_' + i, type: 'LITERAL', value: t.strings[i]});
    }

    return t;
}

exports.parse = parseNameTable;
exports.make = makeNameTable;

},{"../parse":9,"../table":11,"../types":26}],23:[function(_dereq_,module,exports){
// The `OS/2` table contains metrics required in OpenType fonts.
// https://www.microsoft.com/typography/OTSPEC/os2.htm

'use strict';

var parse = _dereq_('../parse');
var table = _dereq_('../table');

var unicodeRanges = [
    {begin: 0x0000, end: 0x007F}, // Basic Latin
    {begin: 0x0080, end: 0x00FF}, // Latin-1 Supplement
    {begin: 0x0100, end: 0x017F}, // Latin Extended-A
    {begin: 0x0180, end: 0x024F}, // Latin Extended-B
    {begin: 0x0250, end: 0x02AF}, // IPA Extensions
    {begin: 0x02B0, end: 0x02FF}, // Spacing Modifier Letters
    {begin: 0x0300, end: 0x036F}, // Combining Diacritical Marks
    {begin: 0x0370, end: 0x03FF}, // Greek and Coptic
    {begin: 0x2C80, end: 0x2CFF}, // Coptic
    {begin: 0x0400, end: 0x04FF}, // Cyrillic
    {begin: 0x0530, end: 0x058F}, // Armenian
    {begin: 0x0590, end: 0x05FF}, // Hebrew
    {begin: 0xA500, end: 0xA63F}, // Vai
    {begin: 0x0600, end: 0x06FF}, // Arabic
    {begin: 0x07C0, end: 0x07FF}, // NKo
    {begin: 0x0900, end: 0x097F}, // Devanagari
    {begin: 0x0980, end: 0x09FF}, // Bengali
    {begin: 0x0A00, end: 0x0A7F}, // Gurmukhi
    {begin: 0x0A80, end: 0x0AFF}, // Gujarati
    {begin: 0x0B00, end: 0x0B7F}, // Oriya
    {begin: 0x0B80, end: 0x0BFF}, // Tamil
    {begin: 0x0C00, end: 0x0C7F}, // Telugu
    {begin: 0x0C80, end: 0x0CFF}, // Kannada
    {begin: 0x0D00, end: 0x0D7F}, // Malayalam
    {begin: 0x0E00, end: 0x0E7F}, // Thai
    {begin: 0x0E80, end: 0x0EFF}, // Lao
    {begin: 0x10A0, end: 0x10FF}, // Georgian
    {begin: 0x1B00, end: 0x1B7F}, // Balinese
    {begin: 0x1100, end: 0x11FF}, // Hangul Jamo
    {begin: 0x1E00, end: 0x1EFF}, // Latin Extended Additional
    {begin: 0x1F00, end: 0x1FFF}, // Greek Extended
    {begin: 0x2000, end: 0x206F}, // General Punctuation
    {begin: 0x2070, end: 0x209F}, // Superscripts And Subscripts
    {begin: 0x20A0, end: 0x20CF}, // Currency Symbol
    {begin: 0x20D0, end: 0x20FF}, // Combining Diacritical Marks For Symbols
    {begin: 0x2100, end: 0x214F}, // Letterlike Symbols
    {begin: 0x2150, end: 0x218F}, // Number Forms
    {begin: 0x2190, end: 0x21FF}, // Arrows
    {begin: 0x2200, end: 0x22FF}, // Mathematical Operators
    {begin: 0x2300, end: 0x23FF}, // Miscellaneous Technical
    {begin: 0x2400, end: 0x243F}, // Control Pictures
    {begin: 0x2440, end: 0x245F}, // Optical Character Recognition
    {begin: 0x2460, end: 0x24FF}, // Enclosed Alphanumerics
    {begin: 0x2500, end: 0x257F}, // Box Drawing
    {begin: 0x2580, end: 0x259F}, // Block Elements
    {begin: 0x25A0, end: 0x25FF}, // Geometric Shapes
    {begin: 0x2600, end: 0x26FF}, // Miscellaneous Symbols
    {begin: 0x2700, end: 0x27BF}, // Dingbats
    {begin: 0x3000, end: 0x303F}, // CJK Symbols And Punctuation
    {begin: 0x3040, end: 0x309F}, // Hiragana
    {begin: 0x30A0, end: 0x30FF}, // Katakana
    {begin: 0x3100, end: 0x312F}, // Bopomofo
    {begin: 0x3130, end: 0x318F}, // Hangul Compatibility Jamo
    {begin: 0xA840, end: 0xA87F}, // Phags-pa
    {begin: 0x3200, end: 0x32FF}, // Enclosed CJK Letters And Months
    {begin: 0x3300, end: 0x33FF}, // CJK Compatibility
    {begin: 0xAC00, end: 0xD7AF}, // Hangul Syllables
    {begin: 0xD800, end: 0xDFFF}, // Non-Plane 0 *
    {begin: 0x10900, end: 0x1091F}, // Phoenicia
    {begin: 0x4E00, end: 0x9FFF}, // CJK Unified Ideographs
    {begin: 0xE000, end: 0xF8FF}, // Private Use Area (plane 0)
    {begin: 0x31C0, end: 0x31EF}, // CJK Strokes
    {begin: 0xFB00, end: 0xFB4F}, // Alphabetic Presentation Forms
    {begin: 0xFB50, end: 0xFDFF}, // Arabic Presentation Forms-A
    {begin: 0xFE20, end: 0xFE2F}, // Combining Half Marks
    {begin: 0xFE10, end: 0xFE1F}, // Vertical Forms
    {begin: 0xFE50, end: 0xFE6F}, // Small Form Variants
    {begin: 0xFE70, end: 0xFEFF}, // Arabic Presentation Forms-B
    {begin: 0xFF00, end: 0xFFEF}, // Halfwidth And Fullwidth Forms
    {begin: 0xFFF0, end: 0xFFFF}, // Specials
    {begin: 0x0F00, end: 0x0FFF}, // Tibetan
    {begin: 0x0700, end: 0x074F}, // Syriac
    {begin: 0x0780, end: 0x07BF}, // Thaana
    {begin: 0x0D80, end: 0x0DFF}, // Sinhala
    {begin: 0x1000, end: 0x109F}, // Myanmar
    {begin: 0x1200, end: 0x137F}, // Ethiopic
    {begin: 0x13A0, end: 0x13FF}, // Cherokee
    {begin: 0x1400, end: 0x167F}, // Unified Canadian Aboriginal Syllabics
    {begin: 0x1680, end: 0x169F}, // Ogham
    {begin: 0x16A0, end: 0x16FF}, // Runic
    {begin: 0x1780, end: 0x17FF}, // Khmer
    {begin: 0x1800, end: 0x18AF}, // Mongolian
    {begin: 0x2800, end: 0x28FF}, // Braille Patterns
    {begin: 0xA000, end: 0xA48F}, // Yi Syllables
    {begin: 0x1700, end: 0x171F}, // Tagalog
    {begin: 0x10300, end: 0x1032F}, // Old Italic
    {begin: 0x10330, end: 0x1034F}, // Gothic
    {begin: 0x10400, end: 0x1044F}, // Deseret
    {begin: 0x1D000, end: 0x1D0FF}, // Byzantine Musical Symbols
    {begin: 0x1D400, end: 0x1D7FF}, // Mathematical Alphanumeric Symbols
    {begin: 0xFF000, end: 0xFFFFD}, // Private Use (plane 15)
    {begin: 0xFE00, end: 0xFE0F}, // Variation Selectors
    {begin: 0xE0000, end: 0xE007F}, // Tags
    {begin: 0x1900, end: 0x194F}, // Limbu
    {begin: 0x1950, end: 0x197F}, // Tai Le
    {begin: 0x1980, end: 0x19DF}, // New Tai Lue
    {begin: 0x1A00, end: 0x1A1F}, // Buginese
    {begin: 0x2C00, end: 0x2C5F}, // Glagolitic
    {begin: 0x2D30, end: 0x2D7F}, // Tifinagh
    {begin: 0x4DC0, end: 0x4DFF}, // Yijing Hexagram Symbols
    {begin: 0xA800, end: 0xA82F}, // Syloti Nagri
    {begin: 0x10000, end: 0x1007F}, // Linear B Syllabary
    {begin: 0x10140, end: 0x1018F}, // Ancient Greek Numbers
    {begin: 0x10380, end: 0x1039F}, // Ugaritic
    {begin: 0x103A0, end: 0x103DF}, // Old Persian
    {begin: 0x10450, end: 0x1047F}, // Shavian
    {begin: 0x10480, end: 0x104AF}, // Osmanya
    {begin: 0x10800, end: 0x1083F}, // Cypriot Syllabary
    {begin: 0x10A00, end: 0x10A5F}, // Kharoshthi
    {begin: 0x1D300, end: 0x1D35F}, // Tai Xuan Jing Symbols
    {begin: 0x12000, end: 0x123FF}, // Cuneiform
    {begin: 0x1D360, end: 0x1D37F}, // Counting Rod Numerals
    {begin: 0x1B80, end: 0x1BBF}, // Sundanese
    {begin: 0x1C00, end: 0x1C4F}, // Lepcha
    {begin: 0x1C50, end: 0x1C7F}, // Ol Chiki
    {begin: 0xA880, end: 0xA8DF}, // Saurashtra
    {begin: 0xA900, end: 0xA92F}, // Kayah Li
    {begin: 0xA930, end: 0xA95F}, // Rejang
    {begin: 0xAA00, end: 0xAA5F}, // Cham
    {begin: 0x10190, end: 0x101CF}, // Ancient Symbols
    {begin: 0x101D0, end: 0x101FF}, // Phaistos Disc
    {begin: 0x102A0, end: 0x102DF}, // Carian
    {begin: 0x1F030, end: 0x1F09F}  // Domino Tiles
];

function getUnicodeRange(unicode) {
    for (var i = 0; i < unicodeRanges.length; i += 1) {
        var range = unicodeRanges[i];
        if (unicode >= range.begin && unicode < range.end) {
            return i;
        }
    }

    return -1;
}

// Parse the OS/2 and Windows metrics `OS/2` table
function parseOS2Table(data, start) {
    var os2 = {};
    var p = new parse.Parser(data, start);
    os2.version = p.parseUShort();
    os2.xAvgCharWidth = p.parseShort();
    os2.usWeightClass = p.parseUShort();
    os2.usWidthClass = p.parseUShort();
    os2.fsType = p.parseUShort();
    os2.ySubscriptXSize = p.parseShort();
    os2.ySubscriptYSize = p.parseShort();
    os2.ySubscriptXOffset = p.parseShort();
    os2.ySubscriptYOffset = p.parseShort();
    os2.ySuperscriptXSize = p.parseShort();
    os2.ySuperscriptYSize = p.parseShort();
    os2.ySuperscriptXOffset = p.parseShort();
    os2.ySuperscriptYOffset = p.parseShort();
    os2.yStrikeoutSize = p.parseShort();
    os2.yStrikeoutPosition = p.parseShort();
    os2.sFamilyClass = p.parseShort();
    os2.panose = [];
    for (var i = 0; i < 10; i++) {
        os2.panose[i] = p.parseByte();
    }

    os2.ulUnicodeRange1 = p.parseULong();
    os2.ulUnicodeRange2 = p.parseULong();
    os2.ulUnicodeRange3 = p.parseULong();
    os2.ulUnicodeRange4 = p.parseULong();
    os2.achVendID = String.fromCharCode(p.parseByte(), p.parseByte(), p.parseByte(), p.parseByte());
    os2.fsSelection = p.parseUShort();
    os2.usFirstCharIndex = p.parseUShort();
    os2.usLastCharIndex = p.parseUShort();
    os2.sTypoAscender = p.parseShort();
    os2.sTypoDescender = p.parseShort();
    os2.sTypoLineGap = p.parseShort();
    os2.usWinAscent = p.parseUShort();
    os2.usWinDescent = p.parseUShort();
    if (os2.version >= 1) {
        os2.ulCodePageRange1 = p.parseULong();
        os2.ulCodePageRange2 = p.parseULong();
    }

    if (os2.version >= 2) {
        os2.sxHeight = p.parseShort();
        os2.sCapHeight = p.parseShort();
        os2.usDefaultChar = p.parseUShort();
        os2.usBreakChar = p.parseUShort();
        os2.usMaxContent = p.parseUShort();
    }

    return os2;
}

function makeOS2Table(options) {
    return new table.Table('OS/2', [
        {name: 'version', type: 'USHORT', value: 0x0003},
        {name: 'xAvgCharWidth', type: 'SHORT', value: 0},
        {name: 'usWeightClass', type: 'USHORT', value: 0},
        {name: 'usWidthClass', type: 'USHORT', value: 0},
        {name: 'fsType', type: 'USHORT', value: 0},
        {name: 'ySubscriptXSize', type: 'SHORT', value: 650},
        {name: 'ySubscriptYSize', type: 'SHORT', value: 699},
        {name: 'ySubscriptXOffset', type: 'SHORT', value: 0},
        {name: 'ySubscriptYOffset', type: 'SHORT', value: 140},
        {name: 'ySuperscriptXSize', type: 'SHORT', value: 650},
        {name: 'ySuperscriptYSize', type: 'SHORT', value: 699},
        {name: 'ySuperscriptXOffset', type: 'SHORT', value: 0},
        {name: 'ySuperscriptYOffset', type: 'SHORT', value: 479},
        {name: 'yStrikeoutSize', type: 'SHORT', value: 49},
        {name: 'yStrikeoutPosition', type: 'SHORT', value: 258},
        {name: 'sFamilyClass', type: 'SHORT', value: 0},
        {name: 'bFamilyType', type: 'BYTE', value: 0},
        {name: 'bSerifStyle', type: 'BYTE', value: 0},
        {name: 'bWeight', type: 'BYTE', value: 0},
        {name: 'bProportion', type: 'BYTE', value: 0},
        {name: 'bContrast', type: 'BYTE', value: 0},
        {name: 'bStrokeVariation', type: 'BYTE', value: 0},
        {name: 'bArmStyle', type: 'BYTE', value: 0},
        {name: 'bLetterform', type: 'BYTE', value: 0},
        {name: 'bMidline', type: 'BYTE', value: 0},
        {name: 'bXHeight', type: 'BYTE', value: 0},
        {name: 'ulUnicodeRange1', type: 'ULONG', value: 0},
        {name: 'ulUnicodeRange2', type: 'ULONG', value: 0},
        {name: 'ulUnicodeRange3', type: 'ULONG', value: 0},
        {name: 'ulUnicodeRange4', type: 'ULONG', value: 0},
        {name: 'achVendID', type: 'CHARARRAY', value: 'XXXX'},
        {name: 'fsSelection', type: 'USHORT', value: 0},
        {name: 'usFirstCharIndex', type: 'USHORT', value: 0},
        {name: 'usLastCharIndex', type: 'USHORT', value: 0},
        {name: 'sTypoAscender', type: 'SHORT', value: 0},
        {name: 'sTypoDescender', type: 'SHORT', value: 0},
        {name: 'sTypoLineGap', type: 'SHORT', value: 0},
        {name: 'usWinAscent', type: 'USHORT', value: 0},
        {name: 'usWinDescent', type: 'USHORT', value: 0},
        {name: 'ulCodePageRange1', type: 'ULONG', value: 0},
        {name: 'ulCodePageRange2', type: 'ULONG', value: 0},
        {name: 'sxHeight', type: 'SHORT', value: 0},
        {name: 'sCapHeight', type: 'SHORT', value: 0},
        {name: 'usDefaultChar', type: 'USHORT', value: 0},
        {name: 'usBreakChar', type: 'USHORT', value: 0},
        {name: 'usMaxContext', type: 'USHORT', value: 0}
    ], options);
}

exports.unicodeRanges = unicodeRanges;
exports.getUnicodeRange = getUnicodeRange;
exports.parse = parseOS2Table;
exports.make = makeOS2Table;

},{"../parse":9,"../table":11}],24:[function(_dereq_,module,exports){
// The `post` table stores additional PostScript information, such as glyph names.
// https://www.microsoft.com/typography/OTSPEC/post.htm

'use strict';

var encoding = _dereq_('../encoding');
var parse = _dereq_('../parse');
var table = _dereq_('../table');

// Parse the PostScript `post` table
function parsePostTable(data, start) {
    var post = {};
    var p = new parse.Parser(data, start);
    var i;
    post.version = p.parseVersion();
    post.italicAngle = p.parseFixed();
    post.underlinePosition = p.parseShort();
    post.underlineThickness = p.parseShort();
    post.isFixedPitch = p.parseULong();
    post.minMemType42 = p.parseULong();
    post.maxMemType42 = p.parseULong();
    post.minMemType1 = p.parseULong();
    post.maxMemType1 = p.parseULong();
    switch (post.version) {
    case 1:
        post.names = encoding.standardNames.slice();
        break;
    case 2:
        post.numberOfGlyphs = p.parseUShort();
        post.glyphNameIndex = new Array(post.numberOfGlyphs);
        for (i = 0; i < post.numberOfGlyphs; i++) {
            post.glyphNameIndex[i] = p.parseUShort();
        }

        post.names = [];
        for (i = 0; i < post.numberOfGlyphs; i++) {
            if (post.glyphNameIndex[i] >= encoding.standardNames.length) {
                var nameLength = p.parseChar();
                post.names.push(p.parseString(nameLength));
            }
        }

        break;
    case 2.5:
        post.numberOfGlyphs = p.parseUShort();
        post.offset = new Array(post.numberOfGlyphs);
        for (i = 0; i < post.numberOfGlyphs; i++) {
            post.offset[i] = p.parseChar();
        }

        break;
    }
    return post;
}

function makePostTable() {
    return new table.Table('post', [
        {name: 'version', type: 'FIXED', value: 0x00030000},
        {name: 'italicAngle', type: 'FIXED', value: 0},
        {name: 'underlinePosition', type: 'FWORD', value: 0},
        {name: 'underlineThickness', type: 'FWORD', value: 0},
        {name: 'isFixedPitch', type: 'ULONG', value: 0},
        {name: 'minMemType42', type: 'ULONG', value: 0},
        {name: 'maxMemType42', type: 'ULONG', value: 0},
        {name: 'minMemType1', type: 'ULONG', value: 0},
        {name: 'maxMemType1', type: 'ULONG', value: 0}
    ]);
}

exports.parse = parsePostTable;
exports.make = makePostTable;

},{"../encoding":4,"../parse":9,"../table":11}],25:[function(_dereq_,module,exports){
// The `sfnt` wrapper provides organization for the tables in the font.
// It is the top-level data structure in a font.
// https://www.microsoft.com/typography/OTSPEC/otff.htm
// Recommendations for creating OpenType Fonts:
// http://www.microsoft.com/typography/otspec140/recom.htm

'use strict';

var check = _dereq_('../check');
var table = _dereq_('../table');

var cmap = _dereq_('./cmap');
var cff = _dereq_('./cff');
var head = _dereq_('./head');
var hhea = _dereq_('./hhea');
var hmtx = _dereq_('./hmtx');
var maxp = _dereq_('./maxp');
var _name = _dereq_('./name');
var os2 = _dereq_('./os2');
var post = _dereq_('./post');

function log2(v) {
    return Math.log(v) / Math.log(2) | 0;
}

function computeCheckSum(bytes) {
    while (bytes.length % 4 !== 0) {
        bytes.push(0);
    }

    var sum = 0;
    for (var i = 0; i < bytes.length; i += 4) {
        sum += (bytes[i] << 24) +
            (bytes[i + 1] << 16) +
            (bytes[i + 2] << 8) +
            (bytes[i + 3]);
    }

    sum %= Math.pow(2, 32);
    return sum;
}

function makeTableRecord(tag, checkSum, offset, length) {
    return new table.Table('Table Record', [
        {name: 'tag', type: 'TAG', value: tag !== undefined ? tag : ''},
        {name: 'checkSum', type: 'ULONG', value: checkSum !== undefined ? checkSum : 0},
        {name: 'offset', type: 'ULONG', value: offset !== undefined ? offset : 0},
        {name: 'length', type: 'ULONG', value: length !== undefined ? length : 0}
    ]);
}

function makeSfntTable(tables) {
    var sfnt = new table.Table('sfnt', [
        {name: 'version', type: 'TAG', value: 'OTTO'},
        {name: 'numTables', type: 'USHORT', value: 0},
        {name: 'searchRange', type: 'USHORT', value: 0},
        {name: 'entrySelector', type: 'USHORT', value: 0},
        {name: 'rangeShift', type: 'USHORT', value: 0}
    ]);
    sfnt.tables = tables;
    sfnt.numTables = tables.length;
    var highestPowerOf2 = Math.pow(2, log2(sfnt.numTables));
    sfnt.searchRange = 16 * highestPowerOf2;
    sfnt.entrySelector = log2(highestPowerOf2);
    sfnt.rangeShift = sfnt.numTables * 16 - sfnt.searchRange;

    var recordFields = [];
    var tableFields = [];

    var offset = sfnt.sizeOf() + (makeTableRecord().sizeOf() * sfnt.numTables);
    while (offset % 4 !== 0) {
        offset += 1;
        tableFields.push({name: 'padding', type: 'BYTE', value: 0});
    }

    for (var i = 0; i < tables.length; i += 1) {
        var t = tables[i];
        check.argument(t.tableName.length === 4, 'Table name' + t.tableName + ' is invalid.');
        var tableLength = t.sizeOf();
        var tableRecord = makeTableRecord(t.tableName, computeCheckSum(t.encode()), offset, tableLength);
        recordFields.push({name: tableRecord.tag + ' Table Record', type: 'TABLE', value: tableRecord});
        tableFields.push({name: t.tableName + ' table', type: 'TABLE', value: t});
        offset += tableLength;
        check.argument(!isNaN(offset), 'Something went wrong calculating the offset.');
        while (offset % 4 !== 0) {
            offset += 1;
            tableFields.push({name: 'padding', type: 'BYTE', value: 0});
        }
    }

    // Table records need to be sorted alphabetically.
    recordFields.sort(function(r1, r2) {
        if (r1.value.tag > r2.value.tag) {
            return 1;
        } else {
            return -1;
        }
    });

    sfnt.fields = sfnt.fields.concat(recordFields);
    sfnt.fields = sfnt.fields.concat(tableFields);
    return sfnt;
}

// Get the metrics for a character. If the string has more than one character
// this function returns metrics for the first available character.
// You can provide optional fallback metrics if no characters are available.
function metricsForChar(font, chars, notFoundMetrics) {
    for (var i = 0; i < chars.length; i += 1) {
        var glyphIndex = font.charToGlyphIndex(chars[i]);
        if (glyphIndex > 0) {
            var glyph = font.glyphs.get(glyphIndex);
            return glyph.getMetrics();
        }
    }

    return notFoundMetrics;
}

function average(vs) {
    var sum = 0;
    for (var i = 0; i < vs.length; i += 1) {
        sum += vs[i];
    }

    return sum / vs.length;
}

// Convert the font object to a SFNT data structure.
// This structure contains all the necessary tables and metadata to create a binary OTF file.
function fontToSfntTable(font) {
    var xMins = [];
    var yMins = [];
    var xMaxs = [];
    var yMaxs = [];
    var advanceWidths = [];
    var leftSideBearings = [];
    var rightSideBearings = [];
    var firstCharIndex;
    var lastCharIndex = 0;
    var ulUnicodeRange1 = 0;
    var ulUnicodeRange2 = 0;
    var ulUnicodeRange3 = 0;
    var ulUnicodeRange4 = 0;

    for (var i = 0; i < font.glyphs.length; i += 1) {
        var glyph = font.glyphs.get(i);
        var unicode = glyph.unicode | 0;
        if (firstCharIndex > unicode || firstCharIndex === null) {
            firstCharIndex = unicode;
        }

        if (lastCharIndex < unicode) {
            lastCharIndex = unicode;
        }

        var position = os2.getUnicodeRange(unicode);
        if (position < 32) {
            ulUnicodeRange1 |= 1 << position;
        } else if (position < 64) {
            ulUnicodeRange2 |= 1 << position - 32;
        } else if (position < 96) {
            ulUnicodeRange3 |= 1 << position - 64;
        } else if (position < 123) {
            ulUnicodeRange4 |= 1 << position - 96;
        } else {
            throw new Error('Unicode ranges bits > 123 are reserved for internal usage');
        }
        // Skip non-important characters.
        if (glyph.name === '.notdef') continue;
        var metrics = glyph.getMetrics();
        xMins.push(metrics.xMin);
        yMins.push(metrics.yMin);
        xMaxs.push(metrics.xMax);
        yMaxs.push(metrics.yMax);
        leftSideBearings.push(metrics.leftSideBearing);
        rightSideBearings.push(metrics.rightSideBearing);
        advanceWidths.push(glyph.advanceWidth);
    }

    var globals = {
        xMin: Math.min.apply(null, xMins),
        yMin: Math.min.apply(null, yMins),
        xMax: Math.max.apply(null, xMaxs),
        yMax: Math.max.apply(null, yMaxs),
        advanceWidthMax: Math.max.apply(null, advanceWidths),
        advanceWidthAvg: average(advanceWidths),
        minLeftSideBearing: Math.min.apply(null, leftSideBearings),
        maxLeftSideBearing: Math.max.apply(null, leftSideBearings),
        minRightSideBearing: Math.min.apply(null, rightSideBearings)
    };
    globals.ascender = font.ascender !== undefined ? font.ascender : globals.yMax;
    globals.descender = font.descender !== undefined ? font.descender : globals.yMin;

    var headTable = head.make({
        unitsPerEm: font.unitsPerEm,
        xMin: globals.xMin,
        yMin: globals.yMin,
        xMax: globals.xMax,
        yMax: globals.yMax
    });

    var hheaTable = hhea.make({
        ascender: globals.ascender,
        descender: globals.descender,
        advanceWidthMax: globals.advanceWidthMax,
        minLeftSideBearing: globals.minLeftSideBearing,
        minRightSideBearing: globals.minRightSideBearing,
        xMaxExtent: globals.maxLeftSideBearing + (globals.xMax - globals.xMin),
        numberOfHMetrics: font.glyphs.length
    });

    var maxpTable = maxp.make(font.glyphs.length);

    var os2Table = os2.make({
        xAvgCharWidth: Math.round(globals.advanceWidthAvg),
        usWeightClass: 500, // Medium FIXME Make this configurable
        usWidthClass: 5, // Medium (normal) FIXME Make this configurable
        usFirstCharIndex: firstCharIndex,
        usLastCharIndex: lastCharIndex,
        ulUnicodeRange1: ulUnicodeRange1,
        ulUnicodeRange2: ulUnicodeRange2,
        ulUnicodeRange3: ulUnicodeRange3,
        ulUnicodeRange4: ulUnicodeRange4,
        // See http://typophile.com/node/13081 for more info on vertical metrics.
        // We get metrics for typical characters (such as "x" for xHeight).
        // We provide some fallback characters if characters are unavailable: their
        // ordering was chosen experimentally.
        sTypoAscender: globals.ascender,
        sTypoDescender: globals.descender,
        sTypoLineGap: 0,
        usWinAscent: globals.ascender,
        usWinDescent: -globals.descender,
        sxHeight: metricsForChar(font, 'xyvw', {yMax: 0}).yMax,
        sCapHeight: metricsForChar(font, 'HIKLEFJMNTZBDPRAGOQSUVWXY', globals).yMax,
        usBreakChar: font.hasChar(' ') ? 32 : 0 // Use space as the break character, if available.
    });

    var hmtxTable = hmtx.make(font.glyphs);
    var cmapTable = cmap.make(font.glyphs);

    var fullName = font.familyName + ' ' + font.styleName;
    var postScriptName = font.familyName.replace(/\s/g, '') + '-' + font.styleName;
    var nameTable = _name.make({
        copyright: font.copyright,
        fontFamily: font.familyName,
        fontSubfamily: font.styleName,
        uniqueID: font.manufacturer + ':' + fullName,
        fullName: fullName,
        version: font.version,
        postScriptName: postScriptName,
        trademark: font.trademark,
        manufacturer: font.manufacturer,
        designer: font.designer,
        description: font.description,
        manufacturerURL: font.manufacturerURL,
        designerURL: font.designerURL,
        license: font.license,
        licenseURL: font.licenseURL,
        preferredFamily: font.familyName,
        preferredSubfamily: font.styleName
    });
    var postTable = post.make();
    var cffTable = cff.make(font.glyphs, {
        version: font.version,
        fullName: fullName,
        familyName: font.familyName,
        weightName: font.styleName,
        postScriptName: postScriptName,
        unitsPerEm: font.unitsPerEm
    });
    // Order the tables according to the the OpenType specification 1.4.
    var tables = [headTable, hheaTable, maxpTable, os2Table, nameTable, cmapTable, postTable, cffTable, hmtxTable];

    var sfntTable = makeSfntTable(tables);

    // Compute the font's checkSum and store it in head.checkSumAdjustment.
    var bytes = sfntTable.encode();
    var checkSum = computeCheckSum(bytes);
    var tableFields = sfntTable.fields;
    var checkSumAdjusted = false;
    for (i = 0; i < tableFields.length; i += 1) {
        if (tableFields[i].name === 'head table') {
            tableFields[i].value.checkSumAdjustment = 0xB1B0AFBA - checkSum;
            checkSumAdjusted = true;
            break;
        }
    }

    if (!checkSumAdjusted) {
        throw new Error('Could not find head table with checkSum to adjust.');
    }

    return sfntTable;
}

exports.computeCheckSum = computeCheckSum;
exports.make = makeSfntTable;
exports.fontToTable = fontToSfntTable;

},{"../check":2,"../table":11,"./cff":12,"./cmap":13,"./head":16,"./hhea":17,"./hmtx":18,"./maxp":21,"./name":22,"./os2":23,"./post":24}],26:[function(_dereq_,module,exports){
// Data types used in the OpenType font file.
// All OpenType fonts use Motorola-style byte ordering (Big Endian)

/* global WeakMap */

'use strict';

var check = _dereq_('./check');

var LIMIT16 = 32768; // The limit at which a 16-bit number switches signs == 2^15
var LIMIT32 = 2147483648; // The limit at which a 32-bit number switches signs == 2 ^ 31

var decode = {};
var encode = {};
var sizeOf = {};

// Return a function that always returns the same value.
function constant(v) {
    return function() {
        return v;
    };
}

// OpenType data types //////////////////////////////////////////////////////

// Convert an 8-bit unsigned integer to a list of 1 byte.
encode.BYTE = function(v) {
    check.argument(v >= 0 && v <= 255, 'Byte value should be between 0 and 255.');
    return [v];
};

sizeOf.BYTE = constant(1);

// Convert a 8-bit signed integer to a list of 1 byte.
encode.CHAR = function(v) {
    return [v.charCodeAt(0)];
};

sizeOf.BYTE = constant(1);

// Convert an ASCII string to a list of bytes.
encode.CHARARRAY = function(v) {
    var b = [];
    for (var i = 0; i < v.length; i += 1) {
        b.push(v.charCodeAt(i));
    }

    return b;
};

sizeOf.CHARARRAY = function(v) {
    return v.length;
};

// Convert a 16-bit unsigned integer to a list of 2 bytes.
encode.USHORT = function(v) {
    return [(v >> 8) & 0xFF, v & 0xFF];
};

sizeOf.USHORT = constant(2);

// Convert a 16-bit signed integer to a list of 2 bytes.
encode.SHORT = function(v) {
    // Two's complement
    if (v >= LIMIT16) {
        v = -(2 * LIMIT16 - v);
    }

    return [(v >> 8) & 0xFF, v & 0xFF];
};

sizeOf.SHORT = constant(2);

// Convert a 24-bit unsigned integer to a list of 3 bytes.
encode.UINT24 = function(v) {
    return [(v >> 16) & 0xFF, (v >> 8) & 0xFF, v & 0xFF];
};

sizeOf.UINT24 = constant(3);

// Convert a 32-bit unsigned integer to a list of 4 bytes.
encode.ULONG = function(v) {
    return [(v >> 24) & 0xFF, (v >> 16) & 0xFF, (v >> 8) & 0xFF, v & 0xFF];
};

sizeOf.ULONG = constant(4);

// Convert a 32-bit unsigned integer to a list of 4 bytes.
encode.LONG = function(v) {
    // Two's complement
    if (v >= LIMIT32) {
        v = -(2 * LIMIT32 - v);
    }

    return [(v >> 24) & 0xFF, (v >> 16) & 0xFF, (v >> 8) & 0xFF, v & 0xFF];
};

sizeOf.LONG = constant(4);

encode.FIXED = encode.ULONG;
sizeOf.FIXED = sizeOf.ULONG;

encode.FWORD = encode.SHORT;
sizeOf.FWORD = sizeOf.SHORT;

encode.UFWORD = encode.USHORT;
sizeOf.UFWORD = sizeOf.USHORT;

// FIXME Implement LONGDATETIME
encode.LONGDATETIME = function() {
    return [0, 0, 0, 0, 0, 0, 0, 0];
};

sizeOf.LONGDATETIME = constant(8);

// Convert a 4-char tag to a list of 4 bytes.
encode.TAG = function(v) {
    check.argument(v.length === 4, 'Tag should be exactly 4 ASCII characters.');
    return [v.charCodeAt(0),
            v.charCodeAt(1),
            v.charCodeAt(2),
            v.charCodeAt(3)];
};

sizeOf.TAG = constant(4);

// CFF data types ///////////////////////////////////////////////////////////

encode.Card8 = encode.BYTE;
sizeOf.Card8 = sizeOf.BYTE;

encode.Card16 = encode.USHORT;
sizeOf.Card16 = sizeOf.USHORT;

encode.OffSize = encode.BYTE;
sizeOf.OffSize = sizeOf.BYTE;

encode.SID = encode.USHORT;
sizeOf.SID = sizeOf.USHORT;

// Convert a numeric operand or charstring number to a variable-size list of bytes.
encode.NUMBER = function(v) {
    if (v >= -107 && v <= 107) {
        return [v + 139];
    } else if (v >= 108 && v <= 1131) {
        v = v - 108;
        return [(v >> 8) + 247, v & 0xFF];
    } else if (v >= -1131 && v <= -108) {
        v = -v - 108;
        return [(v >> 8) + 251, v & 0xFF];
    } else if (v >= -32768 && v <= 32767) {
        return encode.NUMBER16(v);
    } else {
        return encode.NUMBER32(v);
    }
};

sizeOf.NUMBER = function(v) {
    return encode.NUMBER(v).length;
};

// Convert a signed number between -32768 and +32767 to a three-byte value.
// This ensures we always use three bytes, but is not the most compact format.
encode.NUMBER16 = function(v) {
    return [28, (v >> 8) & 0xFF, v & 0xFF];
};

sizeOf.NUMBER16 = constant(2);

// Convert a signed number between -(2^31) and +(2^31-1) to a four-byte value.
// This is useful if you want to be sure you always use four bytes,
// at the expense of wasting a few bytes for smaller numbers.
encode.NUMBER32 = function(v) {
    return [29, (v >> 24) & 0xFF, (v >> 16) & 0xFF, (v >> 8) & 0xFF, v & 0xFF];
};

sizeOf.NUMBER32 = constant(4);

encode.REAL = function(v) {
    var value = v.toString();

    // Some numbers use an epsilon to encode the value. (e.g. JavaScript will store 0.0000001 as 1e-7)
    // This code converts it back to a number without the epsilon.
    var m = /\.(\d*?)(?:9{5,20}|0{5,20})\d{0,2}(?:e(.+)|$)/.exec(value);
    if (m) {
        var epsilon = parseFloat('1e' + ((m[2] ? +m[2] : 0) + m[1].length));
        value = (Math.round(v * epsilon) / epsilon).toString();
    }

    var nibbles = '';
    var i;
    var ii;
    for (i = 0, ii = value.length; i < ii; i += 1) {
        var c = value[i];
        if (c === 'e') {
            nibbles += value[++i] === '-' ? 'c' : 'b';
        } else if (c === '.') {
            nibbles += 'a';
        } else if (c === '-') {
            nibbles += 'e';
        } else {
            nibbles += c;
        }
    }

    nibbles += (nibbles.length & 1) ? 'f' : 'ff';
    var out = [30];
    for (i = 0, ii = nibbles.length; i < ii; i += 2) {
        out.push(parseInt(nibbles.substr(i, 2), 16));
    }

    return out;
};

sizeOf.REAL = function(v) {
    return encode.REAL(v).length;
};

encode.NAME = encode.CHARARRAY;
sizeOf.NAME = sizeOf.CHARARRAY;

encode.STRING = encode.CHARARRAY;
sizeOf.STRING = sizeOf.CHARARRAY;

// Convert a ASCII string to a list of UTF16 bytes.
encode.UTF16 = function(v) {
    var b = [];
    for (var i = 0; i < v.length; i += 1) {
        b.push(0);
        b.push(v.charCodeAt(i));
    }

    return b;
};

sizeOf.UTF16 = function(v) {
    return v.length * 2;
};

// Convert a list of values to a CFF INDEX structure.
// The values should be objects containing name / type / value.
encode.INDEX = function(l) {
    var i;
    //var offset, offsets, offsetEncoder, encodedOffsets, encodedOffset, data,
    //    dataSize, i, v;
    // Because we have to know which data type to use to encode the offsets,
    // we have to go through the values twice: once to encode the data and
    // calculate the offets, then again to encode the offsets using the fitting data type.
    var offset = 1; // First offset is always 1.
    var offsets = [offset];
    var data = [];
    var dataSize = 0;
    for (i = 0; i < l.length; i += 1) {
        var v = encode.OBJECT(l[i]);
        Array.prototype.push.apply(data, v);
        dataSize += v.length;
        offset += v.length;
        offsets.push(offset);
    }

    if (data.length === 0) {
        return [0, 0];
    }

    var encodedOffsets = [];
    var offSize = (1 + Math.floor(Math.log(dataSize) / Math.log(2)) / 8) | 0;
    var offsetEncoder = [undefined, encode.BYTE, encode.USHORT, encode.UINT24, encode.ULONG][offSize];
    for (i = 0; i < offsets.length; i += 1) {
        var encodedOffset = offsetEncoder(offsets[i]);
        Array.prototype.push.apply(encodedOffsets, encodedOffset);
    }

    return Array.prototype.concat(encode.Card16(l.length),
                           encode.OffSize(offSize),
                           encodedOffsets,
                           data);
};

sizeOf.INDEX = function(v) {
    return encode.INDEX(v).length;
};

// Convert an object to a CFF DICT structure.
// The keys should be numeric.
// The values should be objects containing name / type / value.
encode.DICT = function(m) {
    var d = [];
    var keys = Object.keys(m);
    var length = keys.length;

    for (var i = 0; i < length; i += 1) {
        // Object.keys() return string keys, but our keys are always numeric.
        var k = parseInt(keys[i], 0);
        var v = m[k];
        // Value comes before the key.
        d = d.concat(encode.OPERAND(v.value, v.type));
        d = d.concat(encode.OPERATOR(k));
    }

    return d;
};

sizeOf.DICT = function(m) {
    return encode.DICT(m).length;
};

encode.OPERATOR = function(v) {
    if (v < 1200) {
        return [v];
    } else {
        return [12, v - 1200];
    }
};

encode.OPERAND = function(v, type) {
    var d = [];
    if (Array.isArray(type)) {
        for (var i = 0; i < type.length; i += 1) {
            check.argument(v.length === type.length, 'Not enough arguments given for type' + type);
            d = d.concat(encode.OPERAND(v[i], type[i]));
        }
    } else {
        if (type === 'SID') {
            d = d.concat(encode.NUMBER(v));
        } else if (type === 'offset') {
            // We make it easy for ourselves and always encode offsets as
            // 4 bytes. This makes offset calculation for the top dict easier.
            d = d.concat(encode.NUMBER32(v));
        } else if (type === 'number') {
            d = d.concat(encode.NUMBER(v));
        } else if (type === 'real') {
            d = d.concat(encode.REAL(v));
        } else {
            throw new Error('Unknown operand type ' + type);
            // FIXME Add support for booleans
        }
    }

    return d;
};

encode.OP = encode.BYTE;
sizeOf.OP = sizeOf.BYTE;

// memoize charstring encoding using WeakMap if available
var wmm = typeof WeakMap === 'function' && new WeakMap();
// Convert a list of CharString operations to bytes.
encode.CHARSTRING = function(ops) {
    if (wmm && wmm.has(ops)) {
        return wmm.get(ops);
    }

    var d = [];
    var length = ops.length;

    for (var i = 0; i < length; i += 1) {
        var op = ops[i];
        d = d.concat(encode[op.type](op.value));
    }

    if (wmm) {
        wmm.set(ops, d);
    }

    return d;
};

sizeOf.CHARSTRING = function(ops) {
    return encode.CHARSTRING(ops).length;
};

// Utility functions ////////////////////////////////////////////////////////

// Convert an object containing name / type / value to bytes.
encode.OBJECT = function(v) {
    var encodingFunction = encode[v.type];
    check.argument(encodingFunction !== undefined, 'No encoding function for type ' + v.type);
    return encodingFunction(v.value);
};

// Convert a table object to bytes.
// A table contains a list of fields containing the metadata (name, type and default value).
// The table itself has the field values set as attributes.
encode.TABLE = function(table) {
    var d = [];
    var length = table.fields.length;

    for (var i = 0; i < length; i += 1) {
        var field = table.fields[i];
        var encodingFunction = encode[field.type];
        check.argument(encodingFunction !== undefined, 'No encoding function for field type ' + field.type);
        var value = table[field.name];
        if (value === undefined) {
            value = field.value;
        }

        var bytes = encodingFunction(value);
        d = d.concat(bytes);
    }

    return d;
};

// Merge in a list of bytes.
encode.LITERAL = function(v) {
    return v;
};

sizeOf.LITERAL = function(v) {
    return v.length;
};

exports.decode = decode;
exports.encode = encode;
exports.sizeOf = sizeOf;

},{"./check":2}],27:[function(_dereq_,module,exports){
/*!
  * Reqwest! A general purpose XHR connection manager
  * license MIT (c) Dustin Diaz 2014
  * https://github.com/ded/reqwest
  */

!function (name, context, definition) {
  if (typeof module != 'undefined' && module.exports) module.exports = definition()
  else if (typeof define == 'function' && define.amd) define(definition)
  else context[name] = definition()
}('reqwest', this, function () {

  var win = window
    , doc = document
    , httpsRe = /^http/
    , protocolRe = /(^\w+):\/\//
    , twoHundo = /^(20\d|1223)$/ //http://stackoverflow.com/questions/10046972/msie-returns-status-code-of-1223-for-ajax-request
    , byTag = 'getElementsByTagName'
    , readyState = 'readyState'
    , contentType = 'Content-Type'
    , requestedWith = 'X-Requested-With'
    , head = doc[byTag]('head')[0]
    , uniqid = 0
    , callbackPrefix = 'reqwest_' + (+new Date())
    , lastValue // data stored by the most recent JSONP callback
    , xmlHttpRequest = 'XMLHttpRequest'
    , xDomainRequest = 'XDomainRequest'
    , noop = function () {}

    , isArray = typeof Array.isArray == 'function'
        ? Array.isArray
        : function (a) {
            return a instanceof Array
          }

    , defaultHeaders = {
          'contentType': 'application/x-www-form-urlencoded'
        , 'requestedWith': xmlHttpRequest
        , 'accept': {
              '*':  'text/javascript, text/html, application/xml, text/xml, */*'
            , 'xml':  'application/xml, text/xml'
            , 'html': 'text/html'
            , 'text': 'text/plain'
            , 'json': 'application/json, text/javascript'
            , 'js':   'application/javascript, text/javascript'
          }
      }

    , xhr = function(o) {
        // is it x-domain
        if (o['crossOrigin'] === true) {
          var xhr = win[xmlHttpRequest] ? new XMLHttpRequest() : null
          if (xhr && 'withCredentials' in xhr) {
            return xhr
          } else if (win[xDomainRequest]) {
            return new XDomainRequest()
          } else {
            throw new Error('Browser does not support cross-origin requests')
          }
        } else if (win[xmlHttpRequest]) {
          return new XMLHttpRequest()
        } else {
          return new ActiveXObject('Microsoft.XMLHTTP')
        }
      }
    , globalSetupOptions = {
        dataFilter: function (data) {
          return data
        }
      }

  function succeed(r) {
    var protocol = protocolRe.exec(r.url);
    protocol = (protocol && protocol[1]) || window.location.protocol;
    return httpsRe.test(protocol) ? twoHundo.test(r.request.status) : !!r.request.response;
  }

  function handleReadyState(r, success, error) {
    return function () {
      // use _aborted to mitigate against IE err c00c023f
      // (can't read props on aborted request objects)
      if (r._aborted) return error(r.request)
      if (r._timedOut) return error(r.request, 'Request is aborted: timeout')
      if (r.request && r.request[readyState] == 4) {
        r.request.onreadystatechange = noop
        if (succeed(r)) success(r.request)
        else
          error(r.request)
      }
    }
  }

  function setHeaders(http, o) {
    var headers = o['headers'] || {}
      , h

    headers['Accept'] = headers['Accept']
      || defaultHeaders['accept'][o['type']]
      || defaultHeaders['accept']['*']

    var isAFormData = typeof FormData === 'function' && (o['data'] instanceof FormData);
    // breaks cross-origin requests with legacy browsers
    if (!o['crossOrigin'] && !headers[requestedWith]) headers[requestedWith] = defaultHeaders['requestedWith']
    if (!headers[contentType] && !isAFormData) headers[contentType] = o['contentType'] || defaultHeaders['contentType']
    for (h in headers)
      headers.hasOwnProperty(h) && 'setRequestHeader' in http && http.setRequestHeader(h, headers[h])
  }

  function setCredentials(http, o) {
    if (typeof o['withCredentials'] !== 'undefined' && typeof http.withCredentials !== 'undefined') {
      http.withCredentials = !!o['withCredentials']
    }
  }

  function generalCallback(data) {
    lastValue = data
  }

  function urlappend (url, s) {
    return url + (/\?/.test(url) ? '&' : '?') + s
  }

  function handleJsonp(o, fn, err, url) {
    var reqId = uniqid++
      , cbkey = o['jsonpCallback'] || 'callback' // the 'callback' key
      , cbval = o['jsonpCallbackName'] || reqwest.getcallbackPrefix(reqId)
      , cbreg = new RegExp('((^|\\?|&)' + cbkey + ')=([^&]+)')
      , match = url.match(cbreg)
      , script = doc.createElement('script')
      , loaded = 0
      , isIE10 = navigator.userAgent.indexOf('MSIE 10.0') !== -1

    if (match) {
      if (match[3] === '?') {
        url = url.replace(cbreg, '$1=' + cbval) // wildcard callback func name
      } else {
        cbval = match[3] // provided callback func name
      }
    } else {
      url = urlappend(url, cbkey + '=' + cbval) // no callback details, add 'em
    }

    win[cbval] = generalCallback

    script.type = 'text/javascript'
    script.src = url
    script.async = true
    if (typeof script.onreadystatechange !== 'undefined' && !isIE10) {
      // need this for IE due to out-of-order onreadystatechange(), binding script
      // execution to an event listener gives us control over when the script
      // is executed. See http://jaubourg.net/2010/07/loading-script-as-onclick-handler-of.html
      script.htmlFor = script.id = '_reqwest_' + reqId
    }

    script.onload = script.onreadystatechange = function () {
      if ((script[readyState] && script[readyState] !== 'complete' && script[readyState] !== 'loaded') || loaded) {
        return false
      }
      script.onload = script.onreadystatechange = null
      script.onclick && script.onclick()
      // Call the user callback with the last value stored and clean up values and scripts.
      fn(lastValue)
      lastValue = undefined
      head.removeChild(script)
      loaded = 1
    }

    // Add the script to the DOM head
    head.appendChild(script)

    // Enable JSONP timeout
    return {
      abort: function () {
        script.onload = script.onreadystatechange = null
        err({}, 'Request is aborted: timeout', {})
        lastValue = undefined
        head.removeChild(script)
        loaded = 1
      }
    }
  }

  function getRequest(fn, err) {
    var o = this.o
      , method = (o['method'] || 'GET').toUpperCase()
      , url = typeof o === 'string' ? o : o['url']
      // convert non-string objects to query-string form unless o['processData'] is false
      , data = (o['processData'] !== false && o['data'] && typeof o['data'] !== 'string')
        ? reqwest.toQueryString(o['data'])
        : (o['data'] || null)
      , http
      , sendWait = false

    // if we're working on a GET request and we have data then we should append
    // query string to end of URL and not post data
    if ((o['type'] == 'jsonp' || method == 'GET') && data) {
      url = urlappend(url, data)
      data = null
    }

    if (o['type'] == 'jsonp') return handleJsonp(o, fn, err, url)

    // get the xhr from the factory if passed
    // if the factory returns null, fall-back to ours
    http = (o.xhr && o.xhr(o)) || xhr(o)

    http.open(method, url, o['async'] === false ? false : true)
    setHeaders(http, o)
    setCredentials(http, o)
    if (win[xDomainRequest] && http instanceof win[xDomainRequest]) {
        http.onload = fn
        http.onerror = err
        // NOTE: see
        // http://social.msdn.microsoft.com/Forums/en-US/iewebdevelopment/thread/30ef3add-767c-4436-b8a9-f1ca19b4812e
        http.onprogress = function() {}
        sendWait = true
    } else {
      http.onreadystatechange = handleReadyState(this, fn, err)
    }
    o['before'] && o['before'](http)
    if (sendWait) {
      setTimeout(function () {
        http.send(data)
      }, 200)
    } else {
      http.send(data)
    }
    return http
  }

  function Reqwest(o, fn) {
    this.o = o
    this.fn = fn

    init.apply(this, arguments)
  }

  function setType(header) {
    // json, javascript, text/plain, text/html, xml
    if (header.match('json')) return 'json'
    if (header.match('javascript')) return 'js'
    if (header.match('text')) return 'html'
    if (header.match('xml')) return 'xml'
  }

  function init(o, fn) {

    this.url = typeof o == 'string' ? o : o['url']
    this.timeout = null

    // whether request has been fulfilled for purpose
    // of tracking the Promises
    this._fulfilled = false
    // success handlers
    this._successHandler = function(){}
    this._fulfillmentHandlers = []
    // error handlers
    this._errorHandlers = []
    // complete (both success and fail) handlers
    this._completeHandlers = []
    this._erred = false
    this._responseArgs = {}

    var self = this

    fn = fn || function () {}

    if (o['timeout']) {
      this.timeout = setTimeout(function () {
        timedOut()
      }, o['timeout'])
    }

    if (o['success']) {
      this._successHandler = function () {
        o['success'].apply(o, arguments)
      }
    }

    if (o['error']) {
      this._errorHandlers.push(function () {
        o['error'].apply(o, arguments)
      })
    }

    if (o['complete']) {
      this._completeHandlers.push(function () {
        o['complete'].apply(o, arguments)
      })
    }

    function complete (resp) {
      o['timeout'] && clearTimeout(self.timeout)
      self.timeout = null
      while (self._completeHandlers.length > 0) {
        self._completeHandlers.shift()(resp)
      }
    }

    function success (resp) {
      var type = o['type'] || resp && setType(resp.getResponseHeader('Content-Type')) // resp can be undefined in IE
      resp = (type !== 'jsonp') ? self.request : resp
      // use global data filter on response text
      var filteredResponse = globalSetupOptions.dataFilter(resp.responseText, type)
        , r = filteredResponse
      try {
        resp.responseText = r
      } catch (e) {
        // can't assign this in IE<=8, just ignore
      }
      if (r) {
        switch (type) {
        case 'json':
          try {
            resp = win.JSON ? win.JSON.parse(r) : eval('(' + r + ')')
          } catch (err) {
            return error(resp, 'Could not parse JSON in response', err)
          }
          break
        case 'js':
          resp = eval(r)
          break
        case 'html':
          resp = r
          break
        case 'xml':
          resp = resp.responseXML
              && resp.responseXML.parseError // IE trololo
              && resp.responseXML.parseError.errorCode
              && resp.responseXML.parseError.reason
            ? null
            : resp.responseXML
          break
        }
      }

      self._responseArgs.resp = resp
      self._fulfilled = true
      fn(resp)
      self._successHandler(resp)
      while (self._fulfillmentHandlers.length > 0) {
        resp = self._fulfillmentHandlers.shift()(resp)
      }

      complete(resp)
    }

    function timedOut() {
      self._timedOut = true
      self.request.abort()      
    }

    function error(resp, msg, t) {
      resp = self.request
      self._responseArgs.resp = resp
      self._responseArgs.msg = msg
      self._responseArgs.t = t
      self._erred = true
      while (self._errorHandlers.length > 0) {
        self._errorHandlers.shift()(resp, msg, t)
      }
      complete(resp)
    }

    this.request = getRequest.call(this, success, error)
  }

  Reqwest.prototype = {
    abort: function () {
      this._aborted = true
      this.request.abort()
    }

  , retry: function () {
      init.call(this, this.o, this.fn)
    }

    /**
     * Small deviation from the Promises A CommonJs specification
     * http://wiki.commonjs.org/wiki/Promises/A
     */

    /**
     * `then` will execute upon successful requests
     */
  , then: function (success, fail) {
      success = success || function () {}
      fail = fail || function () {}
      if (this._fulfilled) {
        this._responseArgs.resp = success(this._responseArgs.resp)
      } else if (this._erred) {
        fail(this._responseArgs.resp, this._responseArgs.msg, this._responseArgs.t)
      } else {
        this._fulfillmentHandlers.push(success)
        this._errorHandlers.push(fail)
      }
      return this
    }

    /**
     * `always` will execute whether the request succeeds or fails
     */
  , always: function (fn) {
      if (this._fulfilled || this._erred) {
        fn(this._responseArgs.resp)
      } else {
        this._completeHandlers.push(fn)
      }
      return this
    }

    /**
     * `fail` will execute when the request fails
     */
  , fail: function (fn) {
      if (this._erred) {
        fn(this._responseArgs.resp, this._responseArgs.msg, this._responseArgs.t)
      } else {
        this._errorHandlers.push(fn)
      }
      return this
    }
  , 'catch': function (fn) {
      return this.fail(fn)
    }
  }

  function reqwest(o, fn) {
    return new Reqwest(o, fn)
  }

  // normalize newline variants according to spec -> CRLF
  function normalize(s) {
    return s ? s.replace(/\r?\n/g, '\r\n') : ''
  }

  function serial(el, cb) {
    var n = el.name
      , t = el.tagName.toLowerCase()
      , optCb = function (o) {
          // IE gives value="" even where there is no value attribute
          // 'specified' ref: http://www.w3.org/TR/DOM-Level-3-Core/core.html#ID-862529273
          if (o && !o['disabled'])
            cb(n, normalize(o['attributes']['value'] && o['attributes']['value']['specified'] ? o['value'] : o['text']))
        }
      , ch, ra, val, i

    // don't serialize elements that are disabled or without a name
    if (el.disabled || !n) return

    switch (t) {
    case 'input':
      if (!/reset|button|image|file/i.test(el.type)) {
        ch = /checkbox/i.test(el.type)
        ra = /radio/i.test(el.type)
        val = el.value
        // WebKit gives us "" instead of "on" if a checkbox has no value, so correct it here
        ;(!(ch || ra) || el.checked) && cb(n, normalize(ch && val === '' ? 'on' : val))
      }
      break
    case 'textarea':
      cb(n, normalize(el.value))
      break
    case 'select':
      if (el.type.toLowerCase() === 'select-one') {
        optCb(el.selectedIndex >= 0 ? el.options[el.selectedIndex] : null)
      } else {
        for (i = 0; el.length && i < el.length; i++) {
          el.options[i].selected && optCb(el.options[i])
        }
      }
      break
    }
  }

  // collect up all form elements found from the passed argument elements all
  // the way down to child elements; pass a '<form>' or form fields.
  // called with 'this'=callback to use for serial() on each element
  function eachFormElement() {
    var cb = this
      , e, i
      , serializeSubtags = function (e, tags) {
          var i, j, fa
          for (i = 0; i < tags.length; i++) {
            fa = e[byTag](tags[i])
            for (j = 0; j < fa.length; j++) serial(fa[j], cb)
          }
        }

    for (i = 0; i < arguments.length; i++) {
      e = arguments[i]
      if (/input|select|textarea/i.test(e.tagName)) serial(e, cb)
      serializeSubtags(e, [ 'input', 'select', 'textarea' ])
    }
  }

  // standard query string style serialization
  function serializeQueryString() {
    return reqwest.toQueryString(reqwest.serializeArray.apply(null, arguments))
  }

  // { 'name': 'value', ... } style serialization
  function serializeHash() {
    var hash = {}
    eachFormElement.apply(function (name, value) {
      if (name in hash) {
        hash[name] && !isArray(hash[name]) && (hash[name] = [hash[name]])
        hash[name].push(value)
      } else hash[name] = value
    }, arguments)
    return hash
  }

  // [ { name: 'name', value: 'value' }, ... ] style serialization
  reqwest.serializeArray = function () {
    var arr = []
    eachFormElement.apply(function (name, value) {
      arr.push({name: name, value: value})
    }, arguments)
    return arr
  }

  reqwest.serialize = function () {
    if (arguments.length === 0) return ''
    var opt, fn
      , args = Array.prototype.slice.call(arguments, 0)

    opt = args.pop()
    opt && opt.nodeType && args.push(opt) && (opt = null)
    opt && (opt = opt.type)

    if (opt == 'map') fn = serializeHash
    else if (opt == 'array') fn = reqwest.serializeArray
    else fn = serializeQueryString

    return fn.apply(null, args)
  }

  reqwest.toQueryString = function (o, trad) {
    var prefix, i
      , traditional = trad || false
      , s = []
      , enc = encodeURIComponent
      , add = function (key, value) {
          // If value is a function, invoke it and return its value
          value = ('function' === typeof value) ? value() : (value == null ? '' : value)
          s[s.length] = enc(key) + '=' + enc(value)
        }
    // If an array was passed in, assume that it is an array of form elements.
    if (isArray(o)) {
      for (i = 0; o && i < o.length; i++) add(o[i]['name'], o[i]['value'])
    } else {
      // If traditional, encode the "old" way (the way 1.3.2 or older
      // did it), otherwise encode params recursively.
      for (prefix in o) {
        if (o.hasOwnProperty(prefix)) buildParams(prefix, o[prefix], traditional, add)
      }
    }

    // spaces should be + according to spec
    return s.join('&').replace(/%20/g, '+')
  }

  function buildParams(prefix, obj, traditional, add) {
    var name, i, v
      , rbracket = /\[\]$/

    if (isArray(obj)) {
      // Serialize array item.
      for (i = 0; obj && i < obj.length; i++) {
        v = obj[i]
        if (traditional || rbracket.test(prefix)) {
          // Treat each array item as a scalar.
          add(prefix, v)
        } else {
          buildParams(prefix + '[' + (typeof v === 'object' ? i : '') + ']', v, traditional, add)
        }
      }
    } else if (obj && obj.toString() === '[object Object]') {
      // Serialize object item.
      for (name in obj) {
        buildParams(prefix + '[' + name + ']', obj[name], traditional, add)
      }

    } else {
      // Serialize scalar item.
      add(prefix, obj)
    }
  }

  reqwest.getcallbackPrefix = function () {
    return callbackPrefix
  }

  // jQuery and Zepto compatibility, differences can be remapped here so you can call
  // .ajax.compat(options, callback)
  reqwest.compat = function (o, fn) {
    if (o) {
      o['type'] && (o['method'] = o['type']) && delete o['type']
      o['dataType'] && (o['type'] = o['dataType'])
      o['jsonpCallback'] && (o['jsonpCallbackName'] = o['jsonpCallback']) && delete o['jsonpCallback']
      o['jsonp'] && (o['jsonpCallback'] = o['jsonp'])
    }
    return new Reqwest(o, fn)
  }

  reqwest.ajaxSetup = function (options) {
    options = options || {}
    for (var k in options) {
      globalSetupOptions[k] = options[k]
    }
  }

  return reqwest
});

},{}],28:[function(_dereq_,module,exports){

'use strict';

var p5 = _dereq_('./core/core');
_dereq_('./color/p5.Color');
_dereq_('./core/p5.Element');
_dereq_('./typography/p5.Font');
_dereq_('./core/p5.Graphics');
_dereq_('./core/p5.Renderer2D');

_dereq_('./image/p5.Image');
_dereq_('./math/p5.Vector');
_dereq_('./io/p5.TableRow');
_dereq_('./io/p5.Table');
_dereq_('./io/p5.XML');

_dereq_('./color/creating_reading');
_dereq_('./color/setting');
_dereq_('./core/constants');
_dereq_('./utilities/conversion');
_dereq_('./utilities/array_functions');
_dereq_('./utilities/string_functions');
_dereq_('./core/environment');
_dereq_('./image/image');
_dereq_('./image/loading_displaying');
_dereq_('./image/pixels');
_dereq_('./io/files');
_dereq_('./events/keyboard');
_dereq_('./events/acceleration'); //john
_dereq_('./events/mouse');
_dereq_('./utilities/time_date');
_dereq_('./events/touch');
_dereq_('./math/math');
_dereq_('./math/calculation');
_dereq_('./math/random');
_dereq_('./math/noise');
_dereq_('./math/trigonometry');
_dereq_('./core/rendering');
_dereq_('./core/2d_primitives');

_dereq_('./core/attributes');
_dereq_('./core/curves');
_dereq_('./core/vertex');
_dereq_('./core/structure');
_dereq_('./core/transform');
_dereq_('./typography/attributes');
_dereq_('./typography/loading_displaying');

_dereq_('./webgl/p5.RendererGL');
_dereq_('./webgl/p5.Geometry');
_dereq_('./webgl/p5.RendererGL.Retained');
_dereq_('./webgl/p5.RendererGL.Immediate');
_dereq_('./webgl/primitives');
_dereq_('./webgl/loading');
_dereq_('./webgl/p5.Matrix');
_dereq_('./webgl/material');
_dereq_('./webgl/light');
_dereq_('./webgl/shader');
_dereq_('./webgl/camera');
_dereq_('./webgl/interaction');

/**
 * _globalInit
 *
 * TODO: ???
 * if sketch is on window
 * assume "global" mode
 * and instantiate p5 automatically
 * otherwise do nothing
 *
 * @return {Undefined}
 */
var _globalInit = function() {
  if (!window.PHANTOMJS && !window.mocha) {
    // If there is a setup or draw function on the window
    // then instantiate p5 in "global" mode
    if((window.setup && typeof window.setup === 'function') ||
      (window.draw && typeof window.draw === 'function')) {
      new p5();
    }
  }
};

// TODO: ???
if (document.readyState === 'complete') {
  _globalInit();
} else {
  window.addEventListener('load', _globalInit , false);
}

module.exports = p5;
},{"./color/creating_reading":30,"./color/p5.Color":31,"./color/setting":32,"./core/2d_primitives":33,"./core/attributes":34,"./core/constants":36,"./core/core":37,"./core/curves":38,"./core/environment":39,"./core/p5.Element":41,"./core/p5.Graphics":42,"./core/p5.Renderer2D":44,"./core/rendering":45,"./core/structure":47,"./core/transform":48,"./core/vertex":49,"./events/acceleration":50,"./events/keyboard":51,"./events/mouse":52,"./events/touch":53,"./image/image":55,"./image/loading_displaying":56,"./image/p5.Image":57,"./image/pixels":58,"./io/files":59,"./io/p5.Table":60,"./io/p5.TableRow":61,"./io/p5.XML":62,"./math/calculation":63,"./math/math":64,"./math/noise":65,"./math/p5.Vector":66,"./math/random":68,"./math/trigonometry":69,"./typography/attributes":70,"./typography/loading_displaying":71,"./typography/p5.Font":72,"./utilities/array_functions":73,"./utilities/conversion":74,"./utilities/string_functions":75,"./utilities/time_date":76,"./webgl/camera":77,"./webgl/interaction":78,"./webgl/light":79,"./webgl/loading":80,"./webgl/material":81,"./webgl/p5.Geometry":82,"./webgl/p5.Matrix":83,"./webgl/p5.RendererGL":86,"./webgl/p5.RendererGL.Immediate":84,"./webgl/p5.RendererGL.Retained":85,"./webgl/primitives":87,"./webgl/shader":88}],29:[function(_dereq_,module,exports){
/**
 * module Conversion
 * submodule Color Conversion
 * @for p5
 * @requires core
 */

'use strict';

/**
 * Conversions adapted from <http://www.easyrgb.com/math.html>.
 *
 * In these functions, hue is always in the range [0,1); all other components
 * are in the range [0,1]. 'Brightness' and 'value' are used interchangeably.
 */

var p5 = _dereq_('../core/core');
p5.ColorConversion = {};

/**
 * Convert an HSBA array to HSLA.
 */
p5.ColorConversion._hsbaToHSLA = function(hsba) {
  var hue = hsba[0];
  var sat = hsba[1];
  var val = hsba[2];

  // Calculate lightness.
  var li = (2 - sat) * val / 2;

  // Convert saturation.
  if (li !== 0) {
    if (li === 1) {
      sat = 0;
    } else if (li < 0.5) {
      sat = sat / (2 - sat);
    } else {
      sat = sat * val / (2 - li * 2);
    }
  }

  // Hue and alpha stay the same.
  return [hue, sat, li, hsba[3]];
};

/**
 * Convert an HSBA array to RGBA.
 */
p5.ColorConversion._hsbaToRGBA = function(hsba) {
  var hue = hsba[0] * 6;  // We will split hue into 6 sectors.
  var sat = hsba[1];
  var val = hsba[2];

  var RGBA = [];

  if (sat === 0) {
    RGBA = [val, val, val, hsba[3]];  // Return early if grayscale.
  } else {
    var sector = Math.floor(hue);
    var tint1 = val * (1 - sat);
    var tint2 = val * (1 - sat * (hue - sector));
    var tint3 = val * (1 - sat * (1 + sector - hue));
    var red, green, blue;
    if (sector === 1) {  // Yellow to green.
      red = tint2;
      green = val;
      blue = tint1;
    } else if (sector === 2) {  // Green to cyan.
      red = tint1;
      green = val;
      blue = tint3;
    } else if (sector === 3) {  // Cyan to blue.
      red = tint1;
      green = tint2;
      blue = val;
    } else if (sector === 4) {  // Blue to magenta.
      red = tint3;
      green = tint1;
      blue = val;
    } else if (sector === 5) {  // Magenta to red.
      red = val;
      green = tint1;
      blue = tint2;
    } else {  // Red to yellow (sector could be 0 or 6).
      red = val;
      green = tint3;
      blue = tint1;
    }
    RGBA = [red, green, blue, hsba[3]];
  }

  return RGBA;
};

/**
 * Convert an HSLA array to HSBA.
 */
p5.ColorConversion._hslaToHSBA = function(hsla) {
  var hue = hsla[0];
  var sat = hsla[1];
  var li = hsla[2];

  // Calculate brightness.
  var val;
  if (li < 0.5) {
    val = (1 + sat) * li;
  } else {
    val = li + sat - li * sat;
  }

  // Convert saturation.
  sat = 2 * (val - li) / val;

  // Hue and alpha stay the same.
  return [hue, sat, val, hsla[3]];
};

/**
 * Convert an HSLA array to RGBA.
 *
 * We need to change basis from HSLA to something that can be more easily be
 * projected onto RGBA. We will choose hue and brightness as our first two
 * components, and pick a convenient third one ('zest') so that we don't need
 * to calculate formal HSBA saturation.
 */
p5.ColorConversion._hslaToRGBA = function(hsla){
  var hue = hsla[0] * 6;  // We will split hue into 6 sectors.
  var sat = hsla[1];
  var li = hsla[2];

  var RGBA = [];

  if (sat === 0) {
    RGBA = [li, li, li, hsla[3]]; // Return early if grayscale.
  } else {

    // Calculate brightness.
    var val;
    if (li < 0.5) {
      val = (1 + sat) * li;
    } else {
      val = li + sat - li * sat;
    }

    // Define zest.
    var zest = 2 * li - val;

    // Implement projection (project onto green by default).
    var hzvToRGB = function(hue, zest, val) {
      if (hue < 0) {  // Hue must wrap to allow projection onto red and blue.
        hue += 6;
      } else if (hue >= 6) {
        hue -= 6;
      }
      if (hue < 1) {  // Red to yellow (increasing green).
        return (zest + (val - zest) * hue);
      } else if (hue < 3) {  // Yellow to cyan (greatest green).
        return val;
      } else if (hue < 4) {  // Cyan to blue (decreasing green).
        return (zest + (val - zest) * (4 - hue));
      } else {  // Blue to red (least green).
        return zest;
      }
    };

    // Perform projections, offsetting hue as necessary.
    RGBA = [hzvToRGB(hue + 2, zest, val),
            hzvToRGB(hue    , zest, val),
            hzvToRGB(hue - 2, zest, val),
            hsla[3]];
  }

  return RGBA;
};

/**
 * Convert an RGBA array to HSBA.
 */
p5.ColorConversion._rgbaToHSBA = function(rgba) {
  var red = rgba[0];
  var green = rgba[1];
  var blue = rgba[2];

  var val = Math.max(red, green, blue);
  var chroma = val - Math.min(red, green, blue);

  var hue, sat;
  if (chroma === 0) {  // Return early if grayscale.
    hue = 0;
    sat = 0;
  }
  else {
    sat = chroma / val;
    if (red === val) {  // Magenta to yellow.
      hue = (green - blue) / chroma;
    } else if (green === val) { // Yellow to cyan.
      hue = 2 + (blue - red) / chroma;
    } else if (blue === val) {  // Cyan to magenta.
      hue = 4 + (red - green) / chroma;
    }
    if (hue < 0) {  // Confine hue to the interval [0, 1).
      hue += 6;
    } else if (hue >= 6) {
      hue -= 6;
    }
  }

  return [hue / 6, sat, val, rgba[3]];
};

/**
 * Convert an RGBA array to HSLA.
 */
p5.ColorConversion._rgbaToHSLA = function(rgba) {
  var red = rgba[0];
  var green = rgba[1];
  var blue = rgba[2];

  var val = Math.max(red, green, blue);
  var min = Math.min(red, green, blue);
  var li = val + min;  // We will halve this later.
  var chroma = val - min;

  var hue, sat;
  if (chroma === 0) {  // Return early if grayscale.
    hue = 0;
    sat = 0;
  } else {
    if (li < 1) {
      sat = chroma / li;
    } else {
      sat = chroma / (2 - chroma);
    }
    if (red === val) {  // Magenta to yellow.
      hue = (green - blue) / chroma;
    } else if (green === val) {  // Yellow to cyan.
      hue = 2 + (blue - red) / chroma;
    } else if (blue === val) {  // Cyan to magenta.
      hue = 4 + (red - green) / chroma;
    }
    if (hue < 0) {  // Confine hue to the interval [0, 1).
      hue += 6;
    } else if (hue >= 6) {
      hue -= 6;
    }
  }

  return [hue / 6, sat, li / 2, rgba[3]];
};

module.exports = p5.ColorConversion;

},{"../core/core":37}],30:[function(_dereq_,module,exports){
/**
 * @module Color
 * @submodule Creating & Reading
 * @for p5
 * @requires core
 * @requires constants
 */

'use strict';

var p5 = _dereq_('../core/core');
var constants = _dereq_('../core/constants');
_dereq_('./p5.Color');

/**
 * Extracts the alpha value from a color or pixel array.
 *
 * @method alpha
 * @param {Object} obj p5.Color object or pixel array
 * @example
 * <div>
 * <code>
 * noStroke();
 * c = color(0, 126, 255, 102);
 * fill(c);
 * rect(15, 15, 35, 70);
 * value = alpha(c);  // Sets 'value' to 102
 * fill(value);
 * rect(50, 15, 35, 70);
 * </code>
 * </div>
 */
p5.prototype.alpha = function(c) {
  if (c instanceof p5.Color || c instanceof Array) {
    return this.color(c)._getAlpha();
  } else {
    throw new Error('Needs p5.Color or pixel array as argument.');
  }
};

/**
 * Extracts the blue value from a color or pixel array.
 *
 * @method blue
 * @param {Object} obj p5.Color object or pixel array
 * @example
 * <div>
 * <code>
 * c = color(175, 100, 220);  // Define color 'c'
 * fill(c);  // Use color variable 'c' as fill color
 * rect(15, 20, 35, 60);  // Draw left rectangle
 *
 * blueValue = blue(c);  // Get blue in 'c'
 * println(blueValue);  // Prints "220.0"
 * fill(0, 0, blueValue);  // Use 'blueValue' in new fill
 * rect(50, 20, 35, 60);  // Draw right rectangle
 * </code>
 * </div>
 */
p5.prototype.blue = function(c) {
  if (c instanceof p5.Color || c instanceof Array) {
    return this.color(c)._getBlue();
  } else {
    throw new Error('Needs p5.Color or pixel array as argument.');
  }
};

/**
 * Extracts the HSB brightness value from a color or pixel array.
 *
 * @method brightness
 * @param {Object} color p5.Color object or pixel array
 * @example
 * <div>
 * <code>
 * noStroke();
 * colorMode(HSB, 255);
 * c = color(0, 126, 255);
 * fill(c);
 * rect(15, 20, 35, 60);
 * value = brightness(c);  // Sets 'value' to 255
 * fill(value);
 * rect(50, 20, 35, 60);
 * </code>
 * </div>
 */
p5.prototype.brightness = function(c) {
  if (c instanceof p5.Color || c instanceof Array) {
    return this.color(c)._getBrightness();
  } else {
    throw new Error('Needs p5.Color or pixel array as argument.');
  }
};

/**
 * Creates colors for storing in variables of the color datatype. The
 * parameters are interpreted as RGB or HSB values depending on the
 * current colorMode(). The default mode is RGB values from 0 to 255
 * and, therefore, the function call color(255, 204, 0) will return a
 * bright yellow color.
 * <br><br>
 * Note that if only one value is provided to color(), it will be interpreted
 * as a grayscale value. Add a second value, and it will be used for alpha
 * transparency. When three values are specified, they are interpreted as
 * either RGB or HSB values. Adding a fourth value applies alpha
 * transparency. If a single string parameter is provided it will be
 * interpreted as a CSS-compatible color string.
 *
 * Colors are stored as Numbers or Arrays.
 *
 * @method color
 * @param  {Number|String} gray    number specifying value between white
 *                                 and black.
 * @param  {Number}        [alpha] alpha value relative to current color range
 * @return {Array}                 resulting color
 *
 * @example
 * <div>
 * <code>
 * var c = color(255, 204, 0);  // Define color 'c'
 * fill(c);  // Use color variable 'c' as fill color
 * noStroke();  // Don't draw a stroke around shapes
 * rect(30, 20, 55, 55);  // Draw rectangle
 * </code>
 * </div>
 *
 * <div>
 * <code>
 * var c = color(255, 204, 0);  // Define color 'c'
 * fill(c);  // Use color variable 'c' as fill color
 * noStroke();  // Don't draw a stroke around shapes
 * ellipse(25, 25, 80, 80);  // Draw left circle
 *
 * // Using only one value with color()
 * // generates a grayscale value.
 * var c = color(65);  // Update 'c' with grayscale value
 * fill(c);  // Use updated 'c' as fill color
 * ellipse(75, 75, 80, 80);  // Draw right circle
 * </code>
 * </div>
 *
 * <div>
 * <code>
 * // Named SVG & CSS colors may be used,
 * var c = color('magenta');
 * fill(c);  // Use 'c' as fill color
 * noStroke();  // Don't draw a stroke around shapes
 * rect(20, 20, 60, 60);  // Draw rectangle
 * </code>
 * </div>
 *
 * <div>
 * <code>
 * // as can hex color codes:
 * noStroke();  // Don't draw a stroke around shapes
 * var c = color('#0f0');
 * fill(c);  // Use 'c' as fill color
 * rect(0, 10, 45, 80);  // Draw rectangle
 *
 * c = color('#00ff00');
 * fill(c);  // Use updated 'c' as fill color
 * rect(55, 10, 45, 80);  // Draw rectangle
 * </code>
 * </div>
 *
 * <div>
 * <code>
 * // RGB and RGBA color strings are also supported:
 * // these all set to the same color (solid blue)
 * var c;
 * noStroke();  // Don't draw a stroke around shapes
 * c = color('rgb(0,0,255)');
 * fill(c); // Use 'c' as fill color
 * rect(10, 10, 35, 35);  // Draw rectangle
 *
 * c = color('rgb(0%, 0%, 100%)');
 * fill(c); // Use updated 'c' as fill color
 * rect(55, 10, 35, 35);  // Draw rectangle
 *
 * c = color('rgba(0, 0, 255, 1)');
 * fill(c); // Use updated 'c' as fill color
 * rect(10, 55, 35, 35);  // Draw rectangle
 *
 * c = color('rgba(0%, 0%, 100%, 1)');
 * fill(c); // Use updated 'c' as fill color
 * rect(55, 55, 35, 35);  // Draw rectangle
 * </code>
 * </div>
 *
 * <div>
 * <code>
 * // HSL color is also supported and can be specified
 * // by value
 * var c;
 * noStroke();  // Don't draw a stroke around shapes
 * c = color('hsl(160, 100%, 50%)');
 * fill(c);  // Use 'c' as fill color
 * rect(0, 10, 45, 80);  // Draw rectangle
 *
 * c = color('hsla(160, 100%, 50%, 0.5)');
 * fill(c); // Use updated 'c' as fill color
 * rect(55, 10, 45, 80);  // Draw rectangle
 * </code>
 * </div>
 *
 * <div>
 * <code>
 * // HSB color is also supported and can be specified
 * // by value
 * var c;
 * noStroke();  // Don't draw a stroke around shapes
 * c = color('hsb(160, 100%, 50%)');
 * fill(c);  // Use 'c' as fill color
 * rect(0, 10, 45, 80);  // Draw rectangle
 *
 * c = color('hsba(160, 100%, 50%, 0.5)');
 * fill(c); // Use updated 'c' as fill color
 * rect(55, 10, 45, 80);  // Draw rectangle
 * </code>
 * </div>
 *
 * <div>
 * <code>
 * var c;  // Declare color 'c'
 * noStroke();  // Don't draw a stroke around shapes
 *
 * // If no colorMode is specified, then the
 * // default of RGB with scale of 0-255 is used.
 * c = color(50, 55, 100);  // Create a color for 'c'
 * fill(c);  // Use color variable 'c' as fill color
 * rect(0, 10, 45, 80);  // Draw left rect
 *
 * colorMode(HSB, 100);  // Use HSB with scale of 0-100
 * c = color(50, 55, 100);  // Update 'c' with new color
 * fill(c);  // Use updated 'c' as fill color
 * rect(55, 10, 45, 80);  // Draw right rect
 * </code>
 * </div>
 */

/**
 * @method color
 * @param  {Number|String} v1      red or hue value relative to
 *                                 the current color range, or a color string
 * @param  {Number}        v2      green or saturation value
 *                                 relative to the current color range
 * @param  {Number}        v3      blue or brightness value
 *                                 relative to the current color range
 * @param  {Number}        [alpha]
 */

p5.prototype.color = function() {
  if (arguments[0] instanceof p5.Color) {
    return arguments[0];  // Do nothing if argument is already a color object.
  } else if (arguments[0] instanceof Array) {
    if (this instanceof p5.Renderer) {
      return new p5.Color(this, arguments[0]);
    } else {
      return new p5.Color(this._renderer, arguments[0]);
    }
  } else {
    if (this instanceof p5.Renderer) {
      return new p5.Color(this, arguments);
    } else {
      return new p5.Color(this._renderer, arguments);
    }
  }
};

/**
 * Extracts the green value from a color or pixel array.
 *
 * @method green
 * @param {Object} color p5.Color object or pixel array
 * @example
 * <div>
 * <code>
 * c = color(20, 75, 200);  // Define color 'c'
 * fill(c);  // Use color variable 'c' as fill color
 * rect(15, 20, 35, 60);  // Draw left rectangle
 *
 * greenValue = green(c);  // Get green in 'c'
 * println(greenValue);  // Print "75.0"
 * fill(0, greenValue, 0);  // Use 'greenValue' in new fill
 * rect(50, 20, 35, 60);  // Draw right rectangle
 * </code>
 * </div>
 */
p5.prototype.green = function(c) {
  if (c instanceof p5.Color || c instanceof Array) {
    return this.color(c)._getGreen();
  } else {
    throw new Error('Needs p5.Color or pixel array as argument.');
  }
};

/**
 * Extracts the hue value from a color or pixel array.
 *
 * Hue exists in both HSB and HSL. This function will return the
 * HSB-normalized hue when supplied with an HSB color object (or when supplied
 * with a pixel array while the color mode is HSB), but will default to the
 * HSL-normalized hue otherwise. (The values will only be different if the
 * maximum hue setting for each system is different.)
 *
 * @method hue
 * @param {Object} color p5.Color object or pixel array
 * @example
 * <div>
 * <code>
 * noStroke();
 * colorMode(HSB, 255);
 * c = color(0, 126, 255);
 * fill(c);
 * rect(15, 20, 35, 60);
 * value = hue(c);  // Sets 'value' to "0"
 * fill(value);
 * rect(50, 20, 35, 60);
 * </code>
 * </div>
 */
p5.prototype.hue = function(c) {
  if (c instanceof p5.Color || c instanceof Array) {
    return this.color(c)._getHue();
  } else {
    throw new Error('Needs p5.Color or pixel array as argument.');
  }
};

/**
 * Blends two colors to find a third color somewhere between them. The amt
 * parameter is the amount to interpolate between the two values where 0.0
 * equal to the first color, 0.1 is very near the first color, 0.5 is halfway
 * in between, etc. An amount below 0 will be treated as 0. Likewise, amounts
 * above 1 will be capped at 1. This is different from the behavior of lerp(),
 * but necessary because otherwise numbers outside the range will produce
 * strange and unexpected colors.
 * <br><br>
 * The way that colours are interpolated depends on the current color mode.
 *
 * @method lerpColor
 * @param  {Array/Number} c1  interpolate from this color
 * @param  {Array/Number} c2  interpolate to this color
 * @param  {Number}       amt number between 0 and 1
 * @return {Array/Number}     interpolated color
 * @example
 * <div>
 * <code>
 * colorMode(RGB);
 * stroke(255);
 * background(51);
 * from = color(218, 165, 32);
 * to = color(72, 61, 139);
 * colorMode(RGB);  // Try changing to HSB.
 * interA = lerpColor(from, to, .33);
 * interB = lerpColor(from, to, .66);
 * fill(from);
 * rect(10, 20, 20, 60);
 * fill(interA);
 * rect(30, 20, 20, 60);
 * fill(interB);
 * rect(50, 20, 20, 60);
 * fill(to);
 * rect(70, 20, 20, 60);
 * </code>
 * </div>
 */
p5.prototype.lerpColor = function(c1, c2, amt) {
  var mode = this._renderer._colorMode;
  var maxes = this._renderer._colorMaxes;
  var l0, l1, l2, l3;
  var fromArray, toArray;

  if (mode === constants.RGB) {
    fromArray = c1.levels.map(function(level) {
      return level / 255;
    });
    toArray = c2.levels.map(function(level) {
      return level / 255;
    });
  } else if (mode === constants.HSB) {
    c1._getBrightness();  // Cache hsba so it definitely exists.
    c2._getBrightness();
    fromArray = c1.hsba;
    toArray = c2.hsba;
  } else if (mode === constants.HSL) {
    c1._getLightness();  // Cache hsla so it definitely exists.
    c2._getLightness();
    fromArray = c1.hsla;
    toArray = c2.hsla;
  } else {
    throw new Error (mode + 'cannot be used for interpolation.');
  }

  // Prevent extrapolation.
  amt = Math.max(Math.min(amt, 1), 0);

  // Perform interpolation.
  l0 = this.lerp(fromArray[0], toArray[0], amt);
  l1 = this.lerp(fromArray[1], toArray[1], amt);
  l2 = this.lerp(fromArray[2], toArray[2], amt);
  l3 = this.lerp(fromArray[3], toArray[3], amt);

  // Scale components.
  l0 *= maxes[mode][0];
  l1 *= maxes[mode][1];
  l2 *= maxes[mode][2];
  l3 *= maxes[mode][3];

  return this.color(l0, l1, l2, l3);
};

/**
 * Extracts the HSL lightness value from a color or pixel array.
 *
 * @method lightness
 * @param {Object} color p5.Color object or pixel array
 * @example
 * <div>
 * <code>
 * noStroke();
 * colorMode(HSL);
 * c = color(156, 100, 50, 1);
 * fill(c);
 * rect(15, 20, 35, 60);
 * value = lightness(c);  // Sets 'value' to 50
 * fill(value);
 * rect(50, 20, 35, 60);
 * </code>
 * </div>
 */
p5.prototype.lightness = function(c) {
  if (c instanceof p5.Color || c instanceof Array) {
    return this.color(c)._getLightness();
  } else {
    throw new Error('Needs p5.Color or pixel array as argument.');
  }
};

/**
 * Extracts the red value from a color or pixel array.
 *
 * @method red
 * @param {Object} obj p5.Color object or pixel array
 * @example
 * <div>
 * <code>
 * c = color(255, 204, 0);  // Define color 'c'
 * fill(c);  // Use color variable 'c' as fill color
 * rect(15, 20, 35, 60);  // Draw left rectangle
 *
 * redValue = red(c);  // Get red in 'c'
 * println(redValue);  // Print "255.0"
 * fill(redValue, 0, 0);  // Use 'redValue' in new fill
 * rect(50, 20, 35, 60);  // Draw right rectangle
 * </code>
 * </div>
 *
 * <div>
 * <code>
 * colorMode(RGB, 255);
 * var c = color(127, 255, 0);
 * colorMode(RGB, 1);
 * var myColor = red(c);
 * print(myColor);
 * </code>
 * </div>
 */
p5.prototype.red = function(c) {
  if (c instanceof p5.Color || c instanceof Array) {
    return this.color(c)._getRed();
  } else {
    throw new Error('Needs p5.Color or pixel array as argument.');
  }
};

/**
 * Extracts the saturation value from a color or pixel array.
 *
 * Saturation is scaled differently in HSB and HSL. This function will return
 * the HSB saturation when supplied with an HSB color object (or when supplied
 * with a pixel array while the color mode is HSB), but will default to the
 * HSL saturation otherwise.
 *
 * @method saturation
 * @param {Object} color p5.Color object or pixel array
 * @example
 * <div>
 * <code>
 * noStroke();
 * colorMode(HSB, 255);
 * c = color(0, 126, 255);
 * fill(c);
 * rect(15, 20, 35, 60);
 * value = saturation(c);  // Sets 'value' to 126
 * fill(value);
 * rect(50, 20, 35, 60);
 * </code>
 * </div>
 */
p5.prototype.saturation = function(c) {
  if (c instanceof p5.Color || c instanceof Array) {
    return this.color(c)._getSaturation();
  } else {
    throw new Error('Needs p5.Color or pixel array as argument.');
  }
};

module.exports = p5;

},{"../core/constants":36,"../core/core":37,"./p5.Color":31}],31:[function(_dereq_,module,exports){
/**
 * @module Color
 * @submodule Creating & Reading
 * @for p5
 * @requires core
 * @requires constants
 * @requires color_conversion
 */

var p5 = _dereq_('../core/core');
var constants = _dereq_('../core/constants');
var color_conversion = _dereq_('./color_conversion');

/**
 * We define colors to be immutable objects. Each color stores the color mode
 * and level maxes that applied at the time of its construction. These are
 * used to interpret the input arguments and to format the output e.g. when
 * saturation() is requested.
 *
 * Internally we store an array representing the ideal RGBA values in floating
 * point form, normalized from 0 to 1. From this we calculate the closest
 * screen color (RGBA levels from 0 to 255) and expose this to the renderer.
 *
 * We also cache normalized, floating point components of the color in various
 * representations as they are calculated. This is done to prevent repeating a
 * conversion that has already been performed.
 *
 * @class p5.Color
 * @constructor
 */
p5.Color = function(renderer, vals) {

  // Record color mode and maxes at time of construction.
  this.mode = renderer._colorMode;
  this.maxes = renderer._colorMaxes;

  // Calculate normalized RGBA values.
  if (this.mode !== constants.RGB &&
      this.mode !== constants.HSL &&
      this.mode !== constants.HSB) {
    throw new Error(this.mode + ' is an invalid colorMode.');
  } else {
    this._array = p5.Color._parseInputs.apply(renderer, vals);
  }

  // Expose closest screen color.
  this.levels = this._array.map(function(level) {
    return Math.round(level * 255);
  });

  return this;
};

p5.Color.prototype.toString = function() {
  var a = this.levels;
  var alpha = this._array[3];  // String representation uses normalized alpha.
  return 'rgba('+a[0]+','+a[1]+','+a[2]+','+ alpha +')';
};

p5.Color.prototype._getAlpha = function() {
  return this._array[3] * this.maxes[this.mode][3];
};

p5.Color.prototype._getBlue = function() {
  return this._array[2] * this.maxes[constants.RGB][2];
};

p5.Color.prototype._getBrightness = function() {
  if (!this.hsba) {
    this.hsba = color_conversion._rgbaToHSBA(this._array);
  }
  return this.hsba[2] * this.maxes[constants.HSB][2];
};

p5.Color.prototype._getGreen = function() {
  return this._array[1] * this.maxes[constants.RGB][1];
};

/**
 * Hue is the same in HSB and HSL, but the maximum value may be different.
 * This function will return the HSB-normalized saturation when supplied with
 * an HSB color object, but will default to the HSL-normalized saturation
 * otherwise.
 */
p5.Color.prototype._getHue = function() {
  if (this.mode === constants.HSB) {
    if (!this.hsba) {
      this.hsba = color_conversion._rgbaToHSBA(this._array);
    }
    return this.hsba[0] * this.maxes[constants.HSB][0];
  } else {
    if (!this.hsla) {
      this.hsla = color_conversion._rgbaToHSLA(this._array);
    }
    return this.hsla[0] * this.maxes[constants.HSL][0];
  }
};

p5.Color.prototype._getLightness = function() {
  if (!this.hsla) {
    this.hsla = color_conversion._rgbaToHSLA(this._array);
  }
  return this.hsla[2] * this.maxes[constants.HSL][2];
};

p5.Color.prototype._getRed = function() {
  return this._array[0] * this.maxes[constants.RGB][0];
};

/**
 * Saturation is scaled differently in HSB and HSL. This function will return
 * the HSB saturation when supplied with an HSB color object, but will default
 * to the HSL saturation otherwise.
 */
p5.Color.prototype._getSaturation = function() {
  if (this.mode === constants.HSB) {
    if (!this.hsba) {
      this.hsba = color_conversion._rgbaToHSBA(this._array);
    }
    return this.hsba[1] * this.maxes[constants.HSB][1];
  } else {
    if (!this.hsla) {
      this.hsla = color_conversion._rgbaToHSLA(this._array);
    }
    return this.hsla[1] * this.maxes[constants.HSL][1];
  }
};

/**
 * CSS named colors.
 */
var namedColors = {
  aliceblue:             '#f0f8ff',
  antiquewhite:          '#faebd7',
  aqua:                  '#00ffff',
  aquamarine:            '#7fffd4',
  azure:                 '#f0ffff',
  beige:                 '#f5f5dc',
  bisque:                '#ffe4c4',
  black:                 '#000000',
  blanchedalmond:        '#ffebcd',
  blue:                  '#0000ff',
  blueviolet:            '#8a2be2',
  brown:                 '#a52a2a',
  burlywood:             '#deb887',
  cadetblue:             '#5f9ea0',
  chartreuse:            '#7fff00',
  chocolate:             '#d2691e',
  coral:                 '#ff7f50',
  cornflowerblue:        '#6495ed',
  cornsilk:              '#fff8dc',
  crimson:               '#dc143c',
  cyan:                  '#00ffff',
  darkblue:              '#00008b',
  darkcyan:              '#008b8b',
  darkgoldenrod:         '#b8860b',
  darkgray:              '#a9a9a9',
  darkgreen:             '#006400',
  darkgrey:              '#a9a9a9',
  darkkhaki:             '#bdb76b',
  darkmagenta:           '#8b008b',
  darkolivegreen:        '#556b2f',
  darkorange:            '#ff8c00',
  darkorchid:            '#9932cc',
  darkred:               '#8b0000',
  darksalmon:            '#e9967a',
  darkseagreen:          '#8fbc8f',
  darkslateblue:         '#483d8b',
  darkslategray:         '#2f4f4f',
  darkslategrey:         '#2f4f4f',
  darkturquoise:         '#00ced1',
  darkviolet:            '#9400d3',
  deeppink:              '#ff1493',
  deepskyblue:           '#00bfff',
  dimgray:               '#696969',
  dimgrey:               '#696969',
  dodgerblue:            '#1e90ff',
  firebrick:             '#b22222',
  floralwhite:           '#fffaf0',
  forestgreen:           '#228b22',
  fuchsia:               '#ff00ff',
  gainsboro:             '#dcdcdc',
  ghostwhite:            '#f8f8ff',
  gold:                  '#ffd700',
  goldenrod:             '#daa520',
  gray:                  '#808080',
  green:                 '#008000',
  greenyellow:           '#adff2f',
  grey:                  '#808080',
  honeydew:              '#f0fff0',
  hotpink:               '#ff69b4',
  indianred:             '#cd5c5c',
  indigo:                '#4b0082',
  ivory:                 '#fffff0',
  khaki:                 '#f0e68c',
  lavender:              '#e6e6fa',
  lavenderblush:         '#fff0f5',
  lawngreen:             '#7cfc00',
  lemonchiffon:          '#fffacd',
  lightblue:             '#add8e6',
  lightcoral:            '#f08080',
  lightcyan:             '#e0ffff',
  lightgoldenrodyellow:  '#fafad2',
  lightgray:             '#d3d3d3',
  lightgreen:            '#90ee90',
  lightgrey:             '#d3d3d3',
  lightpink:             '#ffb6c1',
  lightsalmon:           '#ffa07a',
  lightseagreen:         '#20b2aa',
  lightskyblue:          '#87cefa',
  lightslategray:        '#778899',
  lightslategrey:        '#778899',
  lightsteelblue:        '#b0c4de',
  lightyellow:           '#ffffe0',
  lime:                  '#00ff00',
  limegreen:             '#32cd32',
  linen:                 '#faf0e6',
  magenta:               '#ff00ff',
  maroon:                '#800000',
  mediumaquamarine:      '#66cdaa',
  mediumblue:            '#0000cd',
  mediumorchid:          '#ba55d3',
  mediumpurple:          '#9370db',
  mediumseagreen:        '#3cb371',
  mediumslateblue:       '#7b68ee',
  mediumspringgreen:     '#00fa9a',
  mediumturquoise:       '#48d1cc',
  mediumvioletred:       '#c71585',
  midnightblue:          '#191970',
  mintcream:             '#f5fffa',
  mistyrose:             '#ffe4e1',
  moccasin:              '#ffe4b5',
  navajowhite:           '#ffdead',
  navy:                  '#000080',
  oldlace:               '#fdf5e6',
  olive:                 '#808000',
  olivedrab:             '#6b8e23',
  orange:                '#ffa500',
  orangered:             '#ff4500',
  orchid:                '#da70d6',
  palegoldenrod:         '#eee8aa',
  palegreen:             '#98fb98',
  paleturquoise:         '#afeeee',
  palevioletred:         '#db7093',
  papayawhip:            '#ffefd5',
  peachpuff:             '#ffdab9',
  peru:                  '#cd853f',
  pink:                  '#ffc0cb',
  plum:                  '#dda0dd',
  powderblue:            '#b0e0e6',
  purple:                '#800080',
  red:                   '#ff0000',
  rosybrown:             '#bc8f8f',
  royalblue:             '#4169e1',
  saddlebrown:           '#8b4513',
  salmon:                '#fa8072',
  sandybrown:            '#f4a460',
  seagreen:              '#2e8b57',
  seashell:              '#fff5ee',
  sienna:                '#a0522d',
  silver:                '#c0c0c0',
  skyblue:               '#87ceeb',
  slateblue:             '#6a5acd',
  slategray:             '#708090',
  slategrey:             '#708090',
  snow:                  '#fffafa',
  springgreen:           '#00ff7f',
  steelblue:             '#4682b4',
  tan:                   '#d2b48c',
  teal:                  '#008080',
  thistle:               '#d8bfd8',
  tomato:                '#ff6347',
  turquoise:             '#40e0d0',
  violet:                '#ee82ee',
  wheat:                 '#f5deb3',
  white:                 '#ffffff',
  whitesmoke:            '#f5f5f5',
  yellow:                '#ffff00',
  yellowgreen:           '#9acd32'
};

/**
 * These regular expressions are used to build up the patterns for matching
 * viable CSS color strings: fragmenting the regexes in this way increases the
 * legibility and comprehensibility of the code.
 *
 * Note that RGB values of .9 are not parsed by IE, but are supported here for
 * color string consistency.
 */
var WHITESPACE = /\s*/;  // Match zero or more whitespace characters.
var INTEGER = /(\d{1,3})/;  // Match integers: 79, 255, etc.
var DECIMAL = /((?:\d+(?:\.\d+)?)|(?:\.\d+))/;  // Match 129.6, 79, .9, etc.
var PERCENT = new RegExp(DECIMAL.source + '%');  // Match 12.9%, 79%, .9%, etc.

/**
 * Full color string patterns. The capture groups are necessary.
 */
var colorPatterns = {
  // Match colors in format #XXX, e.g. #416.
  HEX3: /^#([a-f0-9])([a-f0-9])([a-f0-9])$/i,

  // Match colors in format #XXXXXX, e.g. #b4d455.
  HEX6: /^#([a-f0-9]{2})([a-f0-9]{2})([a-f0-9]{2})$/i,

  // Match colors in format rgb(R, G, B), e.g. rgb(255, 0, 128).
  RGB: new RegExp([
    '^rgb\\(',
    INTEGER.source,
    ',',
    INTEGER.source,
    ',',
    INTEGER.source,
    '\\)$'
  ].join(WHITESPACE.source), 'i'),

  // Match colors in format rgb(R%, G%, B%), e.g. rgb(100%, 0%, 28.9%).
  RGB_PERCENT: new RegExp([
    '^rgb\\(',
    PERCENT.source,
    ',',
    PERCENT.source,
    ',',
    PERCENT.source,
    '\\)$'
  ].join(WHITESPACE.source), 'i'),

  // Match colors in format rgb(R, G, B, A), e.g. rgb(255, 0, 128, 0.25).
  RGBA: new RegExp([
    '^rgba\\(',
    INTEGER.source,
    ',',
    INTEGER.source,
    ',',
    INTEGER.source,
    ',',
    DECIMAL.source,
    '\\)$'
  ].join(WHITESPACE.source), 'i'),

  // Match colors in format rgb(R%, G%, B%, A), e.g. rgb(100%, 0%, 28.9%, 0.5).
  RGBA_PERCENT: new RegExp([
    '^rgba\\(',
    PERCENT.source,
    ',',
    PERCENT.source,
    ',',
    PERCENT.source,
    ',',
    DECIMAL.source,
    '\\)$'
  ].join(WHITESPACE.source), 'i'),

  // Match colors in format hsla(H, S%, L%), e.g. hsl(100, 40%, 28.9%).
  HSL: new RegExp([
    '^hsl\\(',
    INTEGER.source,
    ',',
    PERCENT.source,
    ',',
    PERCENT.source,
    '\\)$'
  ].join(WHITESPACE.source), 'i'),

  // Match colors in format hsla(H, S%, L%, A), e.g. hsla(100, 40%, 28.9%, 0.5).
  HSLA: new RegExp([
    '^hsla\\(',
    INTEGER.source,
    ',',
    PERCENT.source,
    ',',
    PERCENT.source,
    ',',
    DECIMAL.source,
    '\\)$'
  ].join(WHITESPACE.source), 'i'),

  // Match colors in format hsb(H, S%, B%), e.g. hsb(100, 40%, 28.9%).
  HSB: new RegExp([
    '^hsb\\(',
    INTEGER.source,
    ',',
    PERCENT.source,
    ',',
    PERCENT.source,
    '\\)$'
  ].join(WHITESPACE.source), 'i'),

  // Match colors in format hsba(H, S%, B%, A), e.g. hsba(100, 40%, 28.9%, 0.5).
  HSBA: new RegExp([
    '^hsba\\(',
    INTEGER.source,
    ',',
    PERCENT.source,
    ',',
    PERCENT.source,
    ',',
    DECIMAL.source,
    '\\)$'
  ].join(WHITESPACE.source), 'i')
};

/**
 * For a number of different inputs, returns a color formatted as [r, g, b, a]
 * arrays, with each component normalized between 0 and 1.
 *
 * @param {Array-like} args An 'array-like' object that represents a list of
 *                          arguments
 * @return {Array}          a color formatted as [r, g, b, a]
 *                          Example:
 *                          input        ==> output
 *                          g            ==> [g, g, g, 255]
 *                          g,a          ==> [g, g, g, a]
 *                          r, g, b      ==> [r, g, b, 255]
 *                          r, g, b, a   ==> [r, g, b, a]
 *                          [g]          ==> [g, g, g, 255]
 *                          [g, a]       ==> [g, g, g, a]
 *                          [r, g, b]    ==> [r, g, b, 255]
 *                          [r, g, b, a] ==> [r, g, b, a]
 * @example
 * <div>
 * <code>
 * // todo
 * </code>
 * </div>
 */
p5.Color._parseInputs = function() {
  var numArgs = arguments.length;
  var mode = this._colorMode;
  var maxes = this._colorMaxes;
  var results = [];

  if (numArgs >= 3) {  // Argument is a list of component values.

    results[0] = arguments[0] / maxes[mode][0];
    results[1] = arguments[1] / maxes[mode][1];
    results[2] = arguments[2] / maxes[mode][2];

    // Alpha may be undefined, so default it to 100%.
    if (typeof arguments[3] === 'number') {
      results[3] = arguments[3] / maxes[mode][3];
    } else {
      results[3] = 1;
    }

    // Constrain components to the range [0,1].
    results = results.map(function(value) {
      return Math.max(Math.min(value, 1), 0);
    });

    // Convert to RGBA and return.
    if (mode === constants.HSL) {
      return color_conversion._hslaToRGBA(results);
    } else if (mode === constants.HSB) {
      return color_conversion._hsbaToRGBA(results);
    } else {
      return results;
    }

  } else if (numArgs === 1 && typeof arguments[0] === 'string') {

    var str = arguments[0].trim().toLowerCase();

    // Return if string is a named colour.
    if (namedColors[str]) {
      return p5.Color._parseInputs.apply(this, [namedColors[str]]);
    }

    // Try RGBA pattern matching.
    if (colorPatterns.HEX3.test(str)) {  // #rgb
      results = colorPatterns.HEX3.exec(str).slice(1).map(function(color) {
        return parseInt(color + color, 16) / 255;
      });
      results[3] = 1;
      return results;
    } else if (colorPatterns.HEX6.test(str)) {  // #rrggbb
      results = colorPatterns.HEX6.exec(str).slice(1).map(function(color) {
        return parseInt(color, 16) / 255;
      });
      results[3] = 1;
      return results;
    } else if (colorPatterns.RGB.test(str)) {  // rgb(R,G,B)
      results = colorPatterns.RGB.exec(str).slice(1).map(function(color) {
        return color / 255;
      });
      results[3] = 1;
      return results;
    } else if (colorPatterns.RGB_PERCENT.test(str)) {  // rgb(R%,G%,B%)
      results = colorPatterns.RGB_PERCENT.exec(str).slice(1)
        .map(function(color) {
          return parseFloat(color) / 100;
        });
      results[3] = 1;
      return results;
    } else if (colorPatterns.RGBA.test(str)) {  // rgba(R,G,B,A)
      results = colorPatterns.RGBA.exec(str).slice(1)
        .map(function(color, idx) {
          if (idx === 3) {
            return parseFloat(color);
          }
          return color / 255;
        });
      return results;
    } else if (colorPatterns.RGBA_PERCENT.test(str)) {  // rgba(R%,G%,B%,A%)
      results = colorPatterns.RGBA_PERCENT.exec(str).slice(1)
        .map(function(color, idx) {
          if (idx === 3) {
            return parseFloat(color);
          }
          return parseFloat(color) / 100;
        });
      return results;
    }

    // Try HSLA pattern matching.
    if (colorPatterns.HSL.test(str)) {  // hsl(H,S,L)
      results = colorPatterns.HSL.exec(str).slice(1)
        .map(function(color, idx) {
        if (idx === 0) {
          return parseInt(color, 10) / 360;
        }
        return parseInt(color, 10) / 100;
      });
      results[3] = 1;
    } else if (colorPatterns.HSLA.test(str)) {  // hsla(H,S,L,A)
      results = colorPatterns.HSLA.exec(str).slice(1)
        .map(function(color, idx) {
        if (idx === 0) {
          return parseInt(color, 10) / 360;
        }
        else if (idx === 3) {
          return parseFloat(color);
        }
        return parseInt(color, 10) / 100;
      });
    }
    if (results.length) {
      return color_conversion._hslaToRGBA(results);
    }

    // Try HSBA pattern matching.
    if (colorPatterns.HSB.test(str)) {  // hsb(H,S,B)
      results = colorPatterns.HSB.exec(str).slice(1)
        .map(function(color, idx) {
        if (idx === 0) {
          return parseInt(color, 10) / 360;
        }
        return parseInt(color, 10) / 100;
      });
      results[3] = 1;
    } else if (colorPatterns.HSBA.test(str)) {  // hsba(H,S,B,A)
      results = colorPatterns.HSBA.exec(str).slice(1)
        .map(function(color, idx) {
        if (idx === 0) {
          return parseInt(color, 10) / 360;
        }
        else if (idx === 3) {
          return parseFloat(color);
        }
        return parseInt(color, 10) / 100;
      });
    }
    if (results.length) {
      return color_conversion._hsbaToRGBA(results);
    }

    // Input did not match any CSS color pattern: default to white.
    results = [1, 1, 1, 1];

  } else if ((numArgs === 1 || numArgs === 2) &&
              typeof arguments[0] === 'number') {  // 'Grayscale' mode.

    /**
     * For HSB and HSL, interpret the gray level as a brightness/lightness
     * value (they are equivalent when chroma is zero). For RGB, normalize the
     * gray level according to the blue maximum.
     */
    results[0] = arguments[0] / maxes[mode][2];
    results[1] = arguments[0] / maxes[mode][2];
    results[2] = arguments[0] / maxes[mode][2];

    // Alpha may be undefined, so default it to 100%.
    if (typeof arguments[1] === 'number') {
      results[3] = arguments[1] / maxes[mode][3];
    } else {
      results[3] = 1;
    }

    // Constrain components to the range [0,1].
    results = results.map(function(value) {
      return Math.max(Math.min(value, 1), 0);
    });

  } else {
    throw new Error (arguments + 'is not a valid color representation.');
  }

  return results;
};

module.exports = p5.Color;

},{"../core/constants":36,"../core/core":37,"./color_conversion":29}],32:[function(_dereq_,module,exports){
/**
 * @module Color
 * @submodule Setting
 * @for p5
 * @requires core
 * @requires constants
 */

'use strict';

var p5 = _dereq_('../core/core');
var constants = _dereq_('../core/constants');
_dereq_('./p5.Color');

/**
 * The background() function sets the color used for the background of the
 * p5.js canvas. The default background is light gray. This function is
 * typically used within draw() to clear the display window at the beginning
 * of each frame, but it can be used inside setup() to set the background on
 * the first frame of animation or if the background need only be set once.
 *
 * @method background
 * @param {p5.Color} color     any value created by the color() function
 * @param {Number} [a]         opacity of the background
 *
 * @example
 * <div>
 * <code>
 * // Grayscale integer value
 * background(51);
 * </code>
 * </div>
 *
 * <div>
 * <code>
 * // R, G & B integer values
 * background(255, 204, 0);
 * </code>
 * </div>
 *
 * <div>
 * <code>
 * // H, S & B integer values
 * colorMode(HSB);
 * background(255, 204, 100);
 * </code>
 * </div>
 *
 * <div>
 * <code>
 * // Named SVG/CSS color string
 * background('red');
 * </code>
 * </div>
 *
 * <div>
 * <code>
 * // three-digit hexadecimal RGB notation
 * background('#fae');
 * </code>
 * </div>
 *
 * <div>
 * <code>
 * // six-digit hexadecimal RGB notation
 * background('#222222');
 * </code>
 * </div>
 *
 * <div>
 * <code>
 * // integer RGB notation
 * background('rgb(0,255,0)');
 * </code>
 * </div>
 *
 * <div>
 * <code>
 * // integer RGBA notation
 * background('rgba(0,255,0, 0.25)');
 * </code>
 * </div>
 *
 * <div>
 * <code>
 * // percentage RGB notation
 * background('rgb(100%,0%,10%)');
 * </code>
 * </div>
 *
 * <div>
 * <code>
 * // percentage RGBA notation
 * background('rgba(100%,0%,100%,0.5)');
 * </code>
 * </div>
 *
 * <div>
 * <code>
 * // p5 Color object
 * background(color(0, 0, 255));
 * </code>
 * </div>
 */

/**
 * @method background
 * @param {String} colorstring color string, possible formats include: integer
 *                         rgb() or rgba(), percentage rgb() or rgba(),
 *                         3-digit hex, 6-digit hex
 * @param {Number} [a]
 */

/**
 * @method background
 * @param {Number} gray   specifies a value between white and black
 * @param {Number} [a]
 */

/**
 * @method background
 * @param {Number} v1     red or hue value (depending on the current color
 *                        mode)
 * @param {Number} v2     green or saturation value (depending on the current
 *                        color mode)
 * @param {Number} v3     blue or brightness value (depending on the current
 *                        color mode)
 * @param {Number} [a]
 */

/**
 * @method background
 * @param {p5.Image} image     image created with loadImage() or createImage(),
 *                             to set as background
 *                             (must be same size as the sketch window)
 * @param {Number} [a]
 */
p5.prototype.background = function() {
  if (arguments[0] instanceof p5.Image) {
    this.image(arguments[0], 0, 0, this.width, this.height);
  } else {
    this._renderer.background.apply(this._renderer, arguments);
  }
  return this;
};

/**
 * Clears the pixels within a buffer. This function only works on p5.Canvas
 * objects created with the createCanvas() function; it won't work with the
 * main display window. Unlike the main graphics context, pixels in
 * additional graphics areas created with createGraphics() can be entirely
 * or partially transparent. This function clears everything to make all of
 * the pixels 100% transparent.
 *
 * @method clear
 * @example
 * <div>
 * <code>
 * // Clear the screen on mouse press.
 * function setup() {
 *   createCanvas(100, 100);
 * }
 *
 * function draw() {
 *   ellipse(mouseX, mouseY, 20, 20);
 * }
 *
 * function mousePressed() {
 *   clear();
 * }
 * </code>
 * </div>
 */
p5.prototype.clear = function() {
  this._renderer.clear();
  return this;
};

/**
 * colorMode() changes the way p5.js interprets color data. By default, the
 * parameters for fill(), stroke(), background(), and color() are defined by
 * values between 0 and 255 using the RGB color model. This is equivalent to
 * setting colorMode(RGB, 255). Setting colorMode(HSB) lets you use the HSB
 * system instead. By default, this is colorMode(HSB, 360, 100, 100, 1). You
 * can also use HSL.
 * <br><br>
 * Note: existing color objects remember the mode that they were created in,
 * so you can change modes as you like without affecting their appearance.
 *
 * @method colorMode
 * @param {Number|Constant} mode   either RGB or HSB, corresponding to
 *                                 Red/Green/Blue and Hue/Saturation/Brightness
 *                                 (or Lightness)
 * @param {Number|Constant} [max1] range for the red or hue depending on the
 *                                 current color mode, or range for all values
 * @param {Number|Constant} [max2] range for the green or saturation depending
 *                                 on the current color mode
 * @param {Number|Constant} [max3] range for the blue or brightness/lighntess
 *                                 depending on the current color mode
 * @param {Number|Constant} [maxA] range for the alpha
 * @example
 * <div>
 * <code>
 * noStroke();
 * colorMode(RGB, 100);
 * for (i = 0; i < 100; i++) {
 *   for (j = 0; j < 100; j++) {
 *     stroke(i, j, 0);
 *     point(i, j);
 *   }
 * }
 * </code>
 * </div>
 *
 * <div>
 * <code>
 * noStroke();
 * colorMode(HSB, 100);
 * for (i = 0; i < 100; i++) {
 *   for (j = 0; j < 100; j++) {
 *     stroke(i, j, 100);
 *     point(i, j);
 *   }
 * }
 * </code>
 * </div>
 *
 * <div>
 * <code>
 * colorMode(RGB, 255);
 * var c = color(127, 255, 0);
 *
 * colorMode(RGB, 1);
 * var myColor = c._getRed();
 * text(myColor, 10, 10, 80, 80);
 * </code>
 * </div>
 *
 * <div>
 * <code>
 * noFill();
 * colorMode(RGB, 255, 255, 255, 1);
 * background(255);
 *
 * strokeWeight(4);
 * stroke(255, 0 , 10, 0.3);
 * ellipse(40, 40, 50, 50);
 * ellipse(50, 50, 40, 40);
 * </code>
 * </div>
 */
p5.prototype.colorMode = function() {
  if (arguments[0] === constants.RGB ||
      arguments[0] === constants.HSB ||
      arguments[0] === constants.HSL) {

    // Set color mode.
    this._renderer._colorMode = arguments[0];

    // Set color maxes.
    var maxes = this._renderer._colorMaxes[this._renderer._colorMode];
    if (arguments.length === 2) {
      maxes[0] = arguments[1];  // Red
      maxes[1] = arguments[1];  // Green
      maxes[2] = arguments[1];  // Blue
      maxes[3] = arguments[1];  // Alpha
    } else if (arguments.length === 4) {
      maxes[0] = arguments[1];  // Red
      maxes[1] = arguments[2];  // Green
      maxes[2] = arguments[3];  // Blue
    } else if (arguments.length === 5) {
      maxes[0] = arguments[1];  // Red
      maxes[1] = arguments[2];  // Green
      maxes[2] = arguments[3];  // Blue
      maxes[3] = arguments[4];  // Alpha
    }
  }

  return this;
};

/**
 * Sets the color used to fill shapes. For example, if you run
 * fill(204, 102, 0), all subsequent shapes will be filled with orange. This
 * color is either specified in terms of the RGB or HSB color depending on
 * the current colorMode(). (The default color space is RGB, with each value
 * in the range from 0 to 255).
 * <br><br>
 * If a single string argument is provided, RGB, RGBA and Hex CSS color strings
 * and all named color strings are supported. A p5 Color object can also be
 * provided to set the fill color.
 *
 * @method fill
 * @param {Number|Array|String|p5.Color} v1   gray value, red or hue value
 *                                            (depending on the current color
 *                                            mode), or color Array, or CSS
 *                                            color string
 * @param {Number}                       [v2] green or saturation value
 *                                            (depending on the current
 *                                            color mode)
 * @param {Number}                       [v3] blue or brightness value
 *                                            (depending on the current
 *                                            color mode)
 * @param {Number}                       [a]  opacity of the background
 *
 * @example
 * <div>
 * <code>
 * // Grayscale integer value
 * fill(51);
 * rect(20, 20, 60, 60);
 * </code>
 * </div>
 *
 * <div>
 * <code>
 * // R, G & B integer values
 * fill(255, 204, 0);
 * rect(20, 20, 60, 60);
 * </code>
 * </div>
 *
 * <div>
 * <code>
 * // H, S & B integer values
 * colorMode(HSB);
 * fill(255, 204, 100);
 * rect(20, 20, 60, 60);
 * </code>
 * </div>
 *
 * <div>
 * <code>
 * // Named SVG/CSS color string
 * fill('red');
 * rect(20, 20, 60, 60);
 * </code>
 * </div>
 *
 * <div>
 * <code>
 * // three-digit hexadecimal RGB notation
 * fill('#fae');
 * rect(20, 20, 60, 60);
 * </code>
 * </div>
 *
 * <div>
 * <code>
 * // six-digit hexadecimal RGB notation
 * fill('#222222');
 * rect(20, 20, 60, 60);
 * </code>
 * </div>
 *
 * <div>
 * <code>
 * // integer RGB notation
 * fill('rgb(0,255,0)');
 * rect(20, 20, 60, 60);
 * </code>
 * </div>
 *
 * <div>
 * <code>
 * // integer RGBA notation
 * fill('rgba(0,255,0, 0.25)');
 * rect(20, 20, 60, 60);
 * </code>
 * </div>
 *
 * <div>
 * <code>
 * // percentage RGB notation
 * fill('rgb(100%,0%,10%)');
 * rect(20, 20, 60, 60);
 * </code>
 * </div>
 *
 * <div>
 * <code>
 * // percentage RGBA notation
 * fill('rgba(100%,0%,100%,0.5)');
 * rect(20, 20, 60, 60);
 * </code>
 * </div>
 *
 * <div>
 * <code>
 * // p5 Color object
 * fill(color(0, 0, 255));
 * rect(20, 20, 60, 60);
 * </code>
 * </div>
 */
p5.prototype.fill = function() {
  this._renderer._setProperty('_fillSet', true);
  this._renderer._setProperty('_doFill', true);
  this._renderer.fill.apply(this._renderer, arguments);
  return this;
};

/**
 * Disables filling geometry. If both noStroke() and noFill() are called,
 * nothing will be drawn to the screen.
 *
 * @method noFill
 * @example
 * <div>
 * <code>
 * rect(15, 10, 55, 55);
 * noFill();
 * rect(20, 20, 60, 60);
 * </code>
 * </div>
 */
p5.prototype.noFill = function() {
  this._renderer._setProperty('_doFill', false);
  return this;
};

/**
 * Disables drawing the stroke (outline). If both noStroke() and noFill()
 * are called, nothing will be drawn to the screen.
 *
 * @method noStroke
 * @example
 * <div>
 * <code>
 * noStroke();
 * rect(20, 20, 60, 60);
 * </code>
 * </div>
 */
p5.prototype.noStroke = function() {
  this._renderer._setProperty('_doStroke', false);
  return this;
};

/**
 * Sets the color used to draw lines and borders around shapes. This color
 * is either specified in terms of the RGB or HSB color depending on the
 * current colorMode() (the default color space is RGB, with each value in
 * the range from 0 to 255).
 * <br><br>
 * If a single string argument is provided, RGB, RGBA and Hex CSS color
 * strings and all named color strings are supported. A p5 Color object
 * can also be provided to set the stroke color.
 *
 * @method stroke
 * @param {Number|Array|String|p5.Color} v1   gray value, red or hue value
 *                                            (depending on the current color
 *                                            mode), or color Array, or CSS
 *                                            color string
 * @param {Number}                       [v2] green or saturation value
 *                                            (depending on the current
 *                                            color mode)
 * @param {Number}                       [v3] blue or brightness value
 *                                            (depending on the current
 *                                            color mode)
 * @param {Number}                       [a]  opacity of the background
 *
 * @example
 * <div>
 * <code>
 * // Grayscale integer value
 * strokeWeight(4);
 * stroke(51);
 * rect(20, 20, 60, 60);
 * </code>
 * </div>
 *
 * <div>
 * <code>
 * // R, G & B integer values
 * stroke(255, 204, 0);
 * strokeWeight(4);
 * rect(20, 20, 60, 60);
 * </code>
 * </div>
 *
 * <div>
 * <code>
 * // H, S & B integer values
 * colorMode(HSB);
 * strokeWeight(4);
 * stroke(255, 204, 100);
 * rect(20, 20, 60, 60);
 * </code>
 * </div>
 *
 * <div>
 * <code>
 * // Named SVG/CSS color string
 * stroke('red');
 * strokeWeight(4);
 * rect(20, 20, 60, 60);
 * </code>
 * </div>
 *
 * <div>
 * <code>
 * // three-digit hexadecimal RGB notation
 * stroke('#fae');
 * strokeWeight(4);
 * rect(20, 20, 60, 60);
 * </code>
 * </div>
 *
 * <div>
 * <code>
 * // six-digit hexadecimal RGB notation
 * stroke('#222222');
 * strokeWeight(4);
 * rect(20, 20, 60, 60);
 * </code>
 * </div>
 *
 * <div>
 * <code>
 * // integer RGB notation
 * stroke('rgb(0,255,0)');
 * strokeWeight(4);
 * rect(20, 20, 60, 60);
 * </code>
 * </div>
 *
 * <div>
 * <code>
 * // integer RGBA notation
 * stroke('rgba(0,255,0,0.25)');
 * strokeWeight(4);
 * rect(20, 20, 60, 60);
 * </code>
 * </div>
 *
 * <div>
 * <code>
 * // percentage RGB notation
 * stroke('rgb(100%,0%,10%)');
 * strokeWeight(4);
 * rect(20, 20, 60, 60);
 * </code>
 * </div>
 *
 * <div>
 * <code>
 * // percentage RGBA notation
 * stroke('rgba(100%,0%,100%,0.5)');
 * strokeWeight(4);
 * rect(20, 20, 60, 60);
 * </code>
 * </div>
 *
 * <div>
 * <code>
 * // p5 Color object
 * stroke(color(0, 0, 255));
 * strokeWeight(4);
 * rect(20, 20, 60, 60);
 * </code>
 * </div>
 */
p5.prototype.stroke = function() {
  this._renderer._setProperty('_strokeSet', true);
  this._renderer._setProperty('_doStroke', true);
  this._renderer.stroke.apply(this._renderer, arguments);
  return this;
};

module.exports = p5;

},{"../core/constants":36,"../core/core":37,"./p5.Color":31}],33:[function(_dereq_,module,exports){
/**
 * @module Shape
 * @submodule 2D Primitives
 * @for p5
 * @requires core
 * @requires constants
 */

'use strict';

var p5 = _dereq_('./core');
var constants = _dereq_('./constants');

_dereq_('./error_helpers');

/**
 * Draw an arc to the screen. If called with only a, b, c, d, start, and
 * stop, the arc will be drawn as an open pie. If mode is provided, the arc
 * will be drawn either open, as a chord, or as a pie as specified. The
 * origin may be changed with the ellipseMode() function.<br><br>
 * Note that drawing a full circle (ex: 0 to TWO_PI) will appear blank
 * because 0 and TWO_PI are the same position on the unit circle. The
 * best way to handle this is by using the ellipse() function instead
 * to create a closed ellipse, and to use the arc() function
 * only to draw parts of an ellipse.
 *
 * @method arc
 * @param  {Number} a      x-coordinate of the arc's ellipse
 * @param  {Number} b      y-coordinate of the arc's ellipse
 * @param  {Number} c      width of the arc's ellipse by default
 * @param  {Number} d      height of the arc's ellipse by default
 * @param  {Number} start  angle to start the arc, specified in radians
 * @param  {Number} stop   angle to stop the arc, specified in radians
 * @param  {String} [mode] optional parameter to determine the way of drawing
 *                         the arc
 * @return {Object}        the p5 object
 * @example
 * <div>
 * <code>
 * arc(50, 55, 50, 50, 0, HALF_PI);
 * noFill();
 * arc(50, 55, 60, 60, HALF_PI, PI);
 * arc(50, 55, 70, 70, PI, PI+QUARTER_PI);
 * arc(50, 55, 80, 80, PI+QUARTER_PI, TWO_PI);
 * </code>
 * </div>
 *
 * <div>
 * <code>
 * arc(50, 50, 80, 80, 0, PI+QUARTER_PI, OPEN);
 * </code>
 * </div>
 *
 * <div>
 * <code>
 * arc(50, 50, 80, 80, 0, PI+QUARTER_PI, CHORD);
 * </code>
 * </div>
 *
 * <div>
 * <code>
 * arc(50, 50, 80, 80, 0, PI+QUARTER_PI, PIE);
 * </code>
 * </div>
 */
p5.prototype.arc = function(x, y, w, h, start, stop, mode) {
  var args = new Array(arguments.length);
  for (var i = 0; i < args.length; ++i) {
    args[i] = arguments[i];
  }
  if (!this._renderer._doStroke && !this._renderer._doFill) {
    return this;
  }
  if (this._angleMode === constants.DEGREES) {
    start = this.radians(start);
    stop = this.radians(stop);
  }

  // Make all angles positive...
  while (start < 0) {
    start += constants.TWO_PI;
  }
  while (stop < 0) {
    stop += constants.TWO_PI;
  }
  // ...and confine them to the interval [0,TWO_PI).
  start %= constants.TWO_PI;
  stop %= constants.TWO_PI;

  // account for full circle
  if (stop === start) {
    stop += constants.TWO_PI;
  }

  // Adjust angles to counter linear scaling.
  if (start <= constants.HALF_PI) {
    start = Math.atan(w / h * Math.tan(start));
  } else  if (start > constants.HALF_PI && start <= 3 * constants.HALF_PI) {
    start = Math.atan(w / h * Math.tan(start)) + constants.PI;
  } else {
    start = Math.atan(w / h * Math.tan(start)) + constants.TWO_PI;
  }
  if (stop <= constants.HALF_PI) {
    stop = Math.atan(w / h * Math.tan(stop));
  } else  if (stop > constants.HALF_PI && stop <= 3 * constants.HALF_PI) {
    stop = Math.atan(w / h * Math.tan(stop)) + constants.PI;
  } else {
    stop = Math.atan(w / h * Math.tan(stop)) + constants.TWO_PI;
  }

  // Exceed the interval if necessary in order to preserve the size and
  // orientation of the arc.
  if (start > stop) {
    stop += constants.TWO_PI;
  }
  // p5 supports negative width and heights for ellipses
  w = Math.abs(w);
  h = Math.abs(h);
  this._renderer.arc(x, y, w, h, start, stop, mode);
  return this;
};

/**
 * Draws an ellipse (oval) to the screen. An ellipse with equal width and
 * height is a circle. By default, the first two parameters set the location,
 * and the third and fourth parameters set the shape's width and height. If
 * no height is specified, the value of width is used for both the width and
 * height. The origin may be changed with the ellipseMode() function.
 *
 * @method ellipse
 * @param  {Number} x x-coordinate of the ellipse.
 * @param  {Number} y y-coordinate of the ellipse.
 * @param  {Number} w width of the ellipse.
 * @param  {Number} [h] height of the ellipse.
 * @return {p5}       the p5 object
 * @example
 * <div>
 * <code>
 * ellipse(56, 46, 55, 55);
 * </code>
 * </div>
 */
/**
 * @method ellipse
 * @param {Number} x
 * @param {Number} y
 * @param {Number} z z-coordinate of the ellipse
 * @param {Number} w
 * @param {Number} [h]
 * @return {p5}
 */
p5.prototype.ellipse = function() {
  var args = new Array(arguments.length);
  for (var i = 0; i < args.length; ++i) {
    args[i] = arguments[i];
  }
  // Duplicate 3rd argument if onl 3 given.
  if (args.length === 3) {
    args.push(args[2]);
  }
  if(this._renderer.isP3D){
    // p5 supports negative width and heights for ellipses
    if (args[3] < 0){args[3] = Math.abs(args[3]);}
    if (args[4] < 0){args[4] = Math.abs(args[4]);}
  } else {
    // p5 supports negative width and heights for rects
    if (args[2] < 0){args[2] = Math.abs(args[2]);}
    if (args[3] < 0){args[3] = Math.abs(args[3]);}
  }
  if (!this._renderer._doStroke && !this._renderer._doFill) {
    return this;
  }
  if (this._renderer.isP3D){
    this._renderer.ellipse(args);
  } else {
    this._renderer.ellipse(args[0],//x
      args[1],//y
      args[2],//width
      args[3]);//height
  }
  return this;
};
/**
 * Draws a line (a direct path between two points) to the screen. The version
 * of line() with four parameters draws the line in 2D. To color a line, use
 * the stroke() function. A line cannot be filled, therefore the fill()
 * function will not affect the color of a line. 2D lines are drawn with a
 * width of one pixel by default, but this can be changed with the
 * strokeWeight() function.
 *
 * @method line
 * @param  {Number} x1 the x-coordinate of the first point
 * @param  {Number} y1 the y-coordinate of the first point
 * @param  {Number} x2 the x-coordinate of the second point
 * @param  {Number} y2 the y-coordinate of the second point
 * @return {p5}        the p5 object
 * @example
 * <div>
 * <code>
 * line(30, 20, 85, 75);
 * </code>
 * </div>
 *
 * <div>
 * <code>
 * line(30, 20, 85, 20);
 * stroke(126);
 * line(85, 20, 85, 75);
 * stroke(255);
 * line(85, 75, 30, 75);
 * </code>
 * </div>
 */
////commented out original
// p5.prototype.line = function(x1, y1, x2, y2) {
//   if (!this._renderer._doStroke) {
//     return this;
//   }
//   if(this._renderer.isP3D){
//   } else {
//     this._renderer.line(x1, y1, x2, y2);
//   }
// };
p5.prototype.line = function() {
  if (!this._renderer._doStroke) {
    return this;
  }
  var args = new Array(arguments.length);
  for (var i = 0; i < args.length; ++i) {
    args[i] = arguments[i];
  }
  //check whether we should draw a 3d line or 2d
  if(this._renderer.isP3D){
    this._renderer.line(
      args[0],
      args[1],
      args[2],
      args[3],
      args[4],
      args[5]);
  } else {
    this._renderer.line(
      args[0],
      args[1],
      args[2],
      args[3]);
  }
  return this;
};

/**
 * Draws a point, a coordinate in space at the dimension of one pixel.
 * The first parameter is the horizontal value for the point, the second
 * value is the vertical value for the point. The color of the point is
 * determined by the current stroke.
 *
 * @method point
 * @param  {Number} x the x-coordinate
 * @param  {Number} y the y-coordinate
 * @return {p5}       the p5 object
 * @example
 * <div>
 * <code>
 * point(30, 20);
 * point(85, 20);
 * point(85, 75);
 * point(30, 75);
 * </code>
 * </div>
 */
p5.prototype.point = function() {
  if (!this._renderer._doStroke) {
    return this;
  }
  var args = new Array(arguments.length);
  for (var i = 0; i < args.length; ++i) {
    args[i] = arguments[i];
  }
  //check whether we should draw a 3d line or 2d
  if(this._renderer.isP3D){
    this._renderer.point(
      args[0],
      args[1],
      args[2]
      );
  } else {
    this._renderer.point(
      args[0],
      args[1]
    );
  }
  return this;
};


/**
 * Draw a quad. A quad is a quadrilateral, a four sided polygon. It is
 * similar to a rectangle, but the angles between its edges are not
 * constrained to ninety degrees. The first pair of parameters (x1,y1)
 * sets the first vertex and the subsequent pairs should proceed
 * clockwise or counter-clockwise around the defined shape.
 *
 * @method quad
 * @param {Number} x1 the x-coordinate of the first point
 * @param {Number} y1 the y-coordinate of the first point
 * @param {Number} x2 the x-coordinate of the second point
 * @param {Number} y2 the y-coordinate of the second point
 * @param {Number} x3 the x-coordinate of the third point
 * @param {Number} y3 the y-coordinate of the third point
 * @param {Number} x4 the x-coordinate of the fourth point
 * @param {Number} y4 the y-coordinate of the fourth point
 * @return {p5}     the p5 object
 * @example
 * <div>
 * <code>
 * quad(38, 31, 86, 20, 69, 63, 30, 76);
 * </code>
 * </div>
 */
/**
 * @method quad
 * @param {Number} x1
 * @param {Number} y1
 * @param {Number} z1 the z-coordinate of the first point
 * @param {Number} x2
 * @param {Number} y2
 * @param {Number} z2 the z-coordinate of the second point
 * @param {Number} x3
 * @param {Number} y3
 * @param {Number} z3 the z-coordinate of the third point
 * @param {Number} x4
 * @param {Number} y4
 * @param {Number} z4 the z-coordinate of the fourth point
 * @return {p5} the p5 object
 */
p5.prototype.quad = function() {
  if (!this._renderer._doStroke && !this._renderer._doFill) {
    return this;
  }
  var args = new Array(arguments.length);
  for (var i = 0; i < args.length; ++i) {
    args[i] = arguments[i];
  }
  if(this._renderer.isP3D){
    this._renderer.quad(
      args[0],
      args[1],
      args[2],
      args[3],
      args[4],
      args[5],
      args[6],
      args[7],
      args[8],
      args[9],
      args[10],
      args[11]
      );
  } else {
    this._renderer.quad(
     args[0],
     args[1],
     args[2],
     args[3],
     args[4],
     args[5],
     args[6],
    args[7]
    );
  }
  return this;
};

/**
* Draws a rectangle to the screen. A rectangle is a four-sided shape with
* every angle at ninety degrees. By default, the first two parameters set
* the location of the upper-left corner, the third sets the width, and the
* fourth sets the height. The way these parameters are interpreted, however,
* may be changed with the rectMode() function.
* <br><br>
* The fifth, sixth, seventh and eighth parameters, if specified,
* determine corner radius for the top-right, top-left, lower-right and
* lower-left corners, respectively. An omitted corner radius parameter is set
* to the value of the previously specified radius value in the parameter list.
*
* @method rect
* @param  {Number} x  x-coordinate of the rectangle.
* @param  {Number} y  y-coordinate of the rectangle.
* @param  {Number} w  width of the rectangle.
* @param  {Number} h  height of the rectangle.
* @param  {Number} [tl] optional radius of top-left corner.
* @param  {Number} [tr] optional radius of top-right corner.
* @param  {Number} [br] optional radius of bottom-right corner.
* @param  {Number} [bl] optional radius of bottom-left corner.
* @return {p5}          the p5 object.
* @example
* <div>
* <code>
* // Draw a rectangle at location (30, 20) with a width and height of 55.
* rect(30, 20, 55, 55);
* </code>
* </div>
*
* <div>
* <code>
* // Draw a rectangle with rounded corners, each having a radius of 20.
* rect(30, 20, 55, 55, 20);
* </code>
* </div>
*
* <div>
* <code>
* // Draw a rectangle with rounded corners having the following radii:
* // top-left = 20, top-right = 15, bottom-right = 10, bottom-left = 5.
* rect(30, 20, 55, 55, 20, 15, 10, 5);
* </code>
* </div>
*/
/**
* @method rect
* @param  {Number} x
* @param  {Number} y
* @param  {Number} z  z-coordinate of the rectangle.
* @param  {Number} w
* @param  {Number} h
* @param  {Number} [tl]
* @param  {Number} [tr]
* @param  {Number} [br]
* @param  {Number} [bl]
* @return {p5}          the p5 object.
*/
p5.prototype.rect = function () {
  var args = new Array(arguments.length);
  for (var i = 0; i < args.length; ++i) {
    args[i] = arguments[i];
  }
  if(this._renderer.isP3D){
    // p5 supports negative width and heights for rects
    if (args[3] < 0){args[3] = Math.abs(args[3]);}
    if (args[4] < 0){args[4] = Math.abs(args[4]);}
  } else if (this._renderer._rectMode !== constants.CORNERS) {
    // p5 supports negative width and heights for rects
    if (args[2] < 0){args[2] = Math.abs(args[2]);}
    if (args[3] < 0){args[3] = Math.abs(args[3]);}
  }
  if (!this._renderer._doStroke && !this._renderer._doFill) {
    return;
  }
  this._renderer.rect(args);
  return this;
};

/**
* A triangle is a plane created by connecting three points. The first two
* arguments specify the first point, the middle two arguments specify the
* second point, and the last two arguments specify the third point.
*
* @method triangle
* @param  {Number} x1 x-coordinate of the first point
* @param  {Number} y1 y-coordinate of the first point
* @param  {Number} x2 x-coordinate of the second point
* @param  {Number} y2 y-coordinate of the second point
* @param  {Number} x3 x-coordinate of the third point
* @param  {Number} y3 y-coordinate of the third point
* @return {p5}        the p5 object
* @example
* <div>
* <code>
* triangle(30, 75, 58, 20, 86, 75);
* </code>
* </div>
*/
p5.prototype.triangle = function() {

  if (!this._renderer._doStroke && !this._renderer._doFill) {
    return this;
  }
  var args = new Array(arguments.length);
  for (var i = 0; i < args.length; ++i) {
    args[i] = arguments[i];
  }
  if(this._renderer.isP3D){
    this._renderer.triangle(args);
  } else {
    this._renderer.triangle(
      args[0],
      args[1],
      args[2],
      args[3],
      args[4],
      args[5]
    );
  }
  return this;
};

module.exports = p5;

},{"./constants":36,"./core":37,"./error_helpers":40}],34:[function(_dereq_,module,exports){
/**
 * @module Shape
 * @submodule Attributes
 * @for p5
 * @requires core
 * @requires constants
 */

'use strict';

var p5 = _dereq_('./core');
var constants = _dereq_('./constants');

/**
 * Modifies the location from which ellipses are drawn by changing the way
 * in which parameters given to ellipse() are interpreted.
 * <br><br>
 * The default mode is ellipseMode(CENTER), which interprets the first two
 * parameters of ellipse() as the shape's center point, while the third and
 * fourth parameters are its width and height.
 * <br><br>
 * ellipseMode(RADIUS) also uses the first two parameters of ellipse() as
 * the shape's center point, but uses the third and fourth parameters to
 * specify half of the shapes's width and height.
 * <br><br>
 * ellipseMode(CORNER) interprets the first two parameters of ellipse() as
 * the upper-left corner of the shape, while the third and fourth parameters
 * are its width and height.
 * <br><br>
 * ellipseMode(CORNERS) interprets the first two parameters of ellipse() as
 * the location of one corner of the ellipse's bounding box, and the third
 * and fourth parameters as the location of the opposite corner.
 * <br><br>
 * The parameter must be written in ALL CAPS because Javascript is a
 * case-sensitive language.
 *
 * @method ellipseMode
 * @param  {Number/Constant} mode either CENTER, RADIUS, CORNER, or CORNERS
 * @return {p5}                   the p5 object
 * @example
 * <div>
 * <code>
 * ellipseMode(RADIUS);  // Set ellipseMode to RADIUS
 * fill(255);  // Set fill to white
 * ellipse(50, 50, 30, 30);  // Draw white ellipse using RADIUS mode
 *
 * ellipseMode(CENTER);  // Set ellipseMode to CENTER
 * fill(100);  // Set fill to gray
 * ellipse(50, 50, 30, 30);  // Draw gray ellipse using CENTER mode
 * </code>
 * </div>
 *
 * <div>
 * <code>
 * ellipseMode(CORNER);  // Set ellipseMode is CORNER
 * fill(255);  // Set fill to white
 * ellipse(25, 25, 50, 50);  // Draw white ellipse using CORNER mode
 *
 * ellipseMode(CORNERS);  // Set ellipseMode to CORNERS
 * fill(100);  // Set fill to gray
 * ellipse(25, 25, 50, 50);  // Draw gray ellipse using CORNERS mode
 * </code>
 * </div>
 */
p5.prototype.ellipseMode = function(m) {
  if (m === constants.CORNER ||
    m === constants.CORNERS ||
    m === constants.RADIUS ||
    m === constants.CENTER) {
    this._renderer._ellipseMode = m;
  }
  return this;
};

/**
 * Draws all geometry with jagged (aliased) edges. Note that smooth() is
 * active by default, so it is necessary to call noSmooth() to disable
 * smoothing of geometry, images, and fonts.
 *
 * @method noSmooth
 * @return {p5} the p5 object
 * @example
 * <div>
 * <code>
 * background(0);
 * noStroke();
 * smooth();
 * ellipse(30, 48, 36, 36);
 * noSmooth();
 * ellipse(70, 48, 36, 36);
 * </code>
 * </div>
 */
p5.prototype.noSmooth = function() {
  this._renderer.noSmooth();
  return this;
};

/**
 * Modifies the location from which rectangles are drawn by changing the way
 * in which parameters given to rect() are interpreted.
 * <br><br>
 * The default mode is rectMode(CORNER), which interprets the first two
 * parameters of rect() as the upper-left corner of the shape, while the
 * third and fourth parameters are its width and height.
 * <br><br>
 * rectMode(CORNERS) interprets the first two parameters of rect() as the
 * location of one corner, and the third and fourth parameters as the
 * location of the opposite corner.
 * <br><br>
 * rectMode(CENTER) interprets the first two parameters of rect() as the
 * shape's center point, while the third and fourth parameters are its
 * width and height.
 * <br><br>
 * rectMode(RADIUS) also uses the first two parameters of rect() as the
 * shape's center point, but uses the third and fourth parameters to specify
 * half of the shapes's width and height.
 * <br><br>
 * The parameter must be written in ALL CAPS because Javascript is a
 * case-sensitive language.
 *
 * @method rectMode
 * @param  {Number/Constant} mode either CORNER, CORNERS, CENTER, or RADIUS
 * @return {p5}                   the p5 object
 * @example
 * <div>
 * <code>
 * rectMode(CORNER);  // Default rectMode is CORNER
 * fill(255);  // Set fill to white
 * rect(25, 25, 50, 50);  // Draw white rect using CORNER mode
 *
 * rectMode(CORNERS);  // Set rectMode to CORNERS
 * fill(100);  // Set fill to gray
 * rect(25, 25, 50, 50);  // Draw gray rect using CORNERS mode
 * </code>
 * </div>
 *
 * <div>
 * <code>
 * rectMode(RADIUS);  // Set rectMode to RADIUS
 * fill(255);  // Set fill to white
 * rect(50, 50, 30, 30);  // Draw white rect using RADIUS mode
 *
 * rectMode(CENTER);  // Set rectMode to CENTER
 * fill(100);  // Set fill to gray
 * rect(50, 50, 30, 30);  // Draw gray rect using CENTER mode
 * </code>
 * </div>
 */
p5.prototype.rectMode = function(m) {
  if (m === constants.CORNER ||
    m === constants.CORNERS ||
    m === constants.RADIUS ||
    m === constants.CENTER) {
    this._renderer._rectMode = m;
  }
  return this;
};

/**
 * Draws all geometry with smooth (anti-aliased) edges. smooth() will also
 * improve image quality of resized images. Note that smooth() is active by
 * default; noSmooth() can be used to disable smoothing of geometry,
 * images, and fonts.
 *
 * @method smooth
 * @return {p5} the p5 object
 * @example
 * <div>
 * <code>
 * background(0);
 * noStroke();
 * smooth();
 * ellipse(30, 48, 36, 36);
 * noSmooth();
 * ellipse(70, 48, 36, 36);
 * </code>
 * </div>
 */
p5.prototype.smooth = function() {
  this._renderer.smooth();
  return this;
};

/**
 * Sets the style for rendering line endings. These ends are either squared,
 * extended, or rounded, each of which specified with the corresponding
 * parameters: SQUARE, PROJECT, and ROUND. The default cap is ROUND.
 *
 * @method strokeCap
 * @param  {Number/Constant} cap either SQUARE, PROJECT, or ROUND
 * @return {p5}                  the p5 object
 * @example
 * <div>
 * <code>
 * strokeWeight(12.0);
 * strokeCap(ROUND);
 * line(20, 30, 80, 30);
 * strokeCap(SQUARE);
 * line(20, 50, 80, 50);
 * strokeCap(PROJECT);
 * line(20, 70, 80, 70);
 * </code>
 * </div>
 */
p5.prototype.strokeCap = function(cap) {
  if (cap === constants.ROUND ||
    cap === constants.SQUARE ||
    cap === constants.PROJECT) {
    this._renderer.strokeCap(cap);
  }
  return this;
};

/**
 * Sets the style of the joints which connect line segments. These joints
 * are either mitered, beveled, or rounded and specified with the
 * corresponding parameters MITER, BEVEL, and ROUND. The default joint is
 * MITER.
 *
 * @method strokeJoin
 * @param  {Number/Constant} join either MITER, BEVEL, ROUND
 * @return {p5}                   the p5 object
 * @example
 * <div>
 * <code>
 * noFill();
 * strokeWeight(10.0);
 * strokeJoin(MITER);
 * beginShape();
 * vertex(35, 20);
 * vertex(65, 50);
 * vertex(35, 80);
 * endShape();
 * </code>
 * </div>
 *
 * <div>
 * <code>
 * noFill();
 * strokeWeight(10.0);
 * strokeJoin(BEVEL);
 * beginShape();
 * vertex(35, 20);
 * vertex(65, 50);
 * vertex(35, 80);
 * endShape();
 * </code>
 * </div>
 *
 * <div>
 * <code>
 * noFill();
 * strokeWeight(10.0);
 * strokeJoin(ROUND);
 * beginShape();
 * vertex(35, 20);
 * vertex(65, 50);
 * vertex(35, 80);
 * endShape();
 * </code>
 * </div>
 */
p5.prototype.strokeJoin = function(join) {
  if (join === constants.ROUND ||
    join === constants.BEVEL ||
    join === constants.MITER) {
    this._renderer.strokeJoin(join);
  }
  return this;
};

/**
 * Sets the width of the stroke used for lines, points, and the border
 * around shapes. All widths are set in units of pixels.
 *
 * @method strokeWeight
 * @param  {Number} weight the weight (in pixels) of the stroke
 * @return {p5}            the p5 object
 * @example
 * <div>
 * <code>
 * strokeWeight(1);  // Default
 * line(20, 20, 80, 20);
 * strokeWeight(4);  // Thicker
 * line(20, 40, 80, 40);
 * strokeWeight(10);  // Beastly
 * line(20, 70, 80, 70);
 * </code>
 * </div>
 */
p5.prototype.strokeWeight = function(w) {
  this._renderer.strokeWeight(w);
  return this;
};

module.exports = p5;

},{"./constants":36,"./core":37}],35:[function(_dereq_,module,exports){
/**
 * @requires constants
 */

var constants = _dereq_('./constants');

module.exports = {

  modeAdjust: function(a, b, c, d, mode) {
    if (mode === constants.CORNER) {
      return { x: a, y: b, w: c, h: d };
    } else if (mode === constants.CORNERS) {
      return { x: a, y: b, w: c-a, h: d-b };
    } else if (mode === constants.RADIUS) {
      return { x: a-c, y: b-d, w: 2*c, h: 2*d };
    } else if (mode === constants.CENTER) {
      return { x: a-c*0.5, y: b-d*0.5, w: c, h: d };
    }
  },

  arcModeAdjust: function(a, b, c, d, mode) {
    if (mode === constants.CORNER) {
      return { x: a+c*0.5, y: b+d*0.5, w: c, h: d };
    } else if (mode === constants.CORNERS) {
      return { x: a, y: b, w: c+a, h: d+b };
    } else if (mode === constants.RADIUS) {
      return { x: a, y: b, w: 2*c, h: 2*d };
    } else if (mode === constants.CENTER) {
      return { x: a, y: b, w: c, h: d };
    }
  }

};


},{"./constants":36}],36:[function(_dereq_,module,exports){
/**
 * @module Constants
 * @submodule Constants
 * @for p5
 */

var PI = Math.PI;

module.exports = {

  // GRAPHICS RENDERER
  P2D: 'p2d',
  WEBGL: 'webgl',

  // ENVIRONMENT
  ARROW: 'default',
  CROSS: 'crosshair',
  HAND: 'pointer',
  MOVE: 'move',
  TEXT: 'text',
  WAIT: 'wait',

  // TRIGONOMETRY

  /**
   * HALF_PI is a mathematical constant with the value
   * 1.57079632679489661923. It is half the ratio of the
   * circumference of a circle to its diameter. It is useful in
   * combination with the trigonometric functions sin() and cos().
   *
   * @property HALF_PI
   *
   * @example
   * <div><code>
   * arc(50, 50, 80, 80, 0, HALF_PI);
   * </code></div>
   *
   */
  HALF_PI: PI / 2,
  /**
   * PI is a mathematical constant with the value
   * 3.14159265358979323846. It is the ratio of the circumference
   * of a circle to its diameter. It is useful in combination with
   * the trigonometric functions sin() and cos().
   *
   * @property PI
   *
   * @example
   * <div><code>
   * arc(50, 50, 80, 80, 0, PI);
   * </code></div>
   */
  PI: PI,
  /**
   * QUARTER_PI is a mathematical constant with the value 0.7853982.
   * It is one quarter the ratio of the circumference of a circle to
   * its diameter. It is useful in combination with the trigonometric
   * functions sin() and cos().
   *
   * @property QUARTER_PI
   *
   * @example
   * <div><code>
   * arc(50, 50, 80, 80, 0, QUARTER_PI);
   * </code></div>
   *
   */
  QUARTER_PI: PI / 4,
  /**
   * TAU is an alias for TWO_PI, a mathematical constant with the
   * value 6.28318530717958647693. It is twice the ratio of the
   * circumference of a circle to its diameter. It is useful in
   * combination with the trigonometric functions sin() and cos().
   *
   * @property TAU
   *
   * @example
   * <div><code>
   * arc(50, 50, 80, 80, 0, TAU);
   * </code></div>
   *
   */
  TAU: PI * 2,
  /**
   * TWO_PI is a mathematical constant with the value
   * 6.28318530717958647693. It is twice the ratio of the
   * circumference of a circle to its diameter. It is useful in
   * combination with the trigonometric functions sin() and cos().
   *
   * @property TWO_PI
   *
   * @example
   * <div><code>
   * arc(50, 50, 80, 80, 0, TWO_PI);
   * </code></div>
   *
   */
  TWO_PI: PI * 2,
  DEGREES: 'degrees',
  RADIANS: 'radians',

  // SHAPE
  CORNER: 'corner',
  CORNERS: 'corners',
  RADIUS: 'radius',
  RIGHT: 'right',
  LEFT: 'left',
  CENTER: 'center',
  TOP: 'top',
  BOTTOM: 'bottom',
  BASELINE: 'alphabetic',
  POINTS: 0x0000,
  LINES: 0x0001,
  LINE_STRIP: 0x0003,
  LINE_LOOP: 0x0002,
  TRIANGLES: 0x0004,
  TRIANGLE_FAN: 0x0006,
  TRIANGLE_STRIP: 0x0005,
  QUADS: 'quads',
  QUAD_STRIP: 'quad_strip',
  CLOSE: 'close',
  OPEN: 'open',
  CHORD: 'chord',
  PIE: 'pie',
  PROJECT: 'square', // PEND: careful this is counterintuitive
  SQUARE: 'butt',
  ROUND: 'round',
  BEVEL: 'bevel',
  MITER: 'miter',

  // COLOR
  RGB: 'rgb',
  HSB: 'hsb',
  HSL: 'hsl',

  // DOM EXTENSION
  AUTO: 'auto',

  // INPUT
  ALT: 18,
  BACKSPACE: 8,
  CONTROL: 17,
  DELETE: 46,
  DOWN_ARROW: 40,
  ENTER: 13,
  ESCAPE: 27,
  LEFT_ARROW: 37,
  OPTION: 18,
  RETURN: 13,
  RIGHT_ARROW: 39,
  SHIFT: 16,
  TAB: 9,
  UP_ARROW: 38,

  // RENDERING
  BLEND: 'normal',
  ADD: 'lighter',
  //ADD: 'add', //
  //SUBTRACT: 'subtract', //
  DARKEST: 'darken',
  LIGHTEST: 'lighten',
  DIFFERENCE: 'difference',
  EXCLUSION: 'exclusion',
  MULTIPLY: 'multiply',
  SCREEN: 'screen',
  REPLACE: 'source-over',
  OVERLAY: 'overlay',
  HARD_LIGHT: 'hard-light',
  SOFT_LIGHT: 'soft-light',
  DODGE: 'color-dodge',
  BURN: 'color-burn',

  // FILTERS
  THRESHOLD: 'threshold',
  GRAY: 'gray',
  OPAQUE: 'opaque',
  INVERT: 'invert',
  POSTERIZE: 'posterize',
  DILATE: 'dilate',
  ERODE: 'erode',
  BLUR: 'blur',

  // TYPOGRAPHY
  NORMAL: 'normal',
  ITALIC: 'italic',
  BOLD: 'bold',

  // TYPOGRAPHY-INTERNAL
  _DEFAULT_TEXT_FILL: '#000000',
  _DEFAULT_LEADMULT: 1.25,
  _CTX_MIDDLE: 'middle',

  // VERTICES
  LINEAR: 'linear',
  QUADRATIC: 'quadratic',
  BEZIER: 'bezier',
  CURVE: 'curve',

  // DEFAULTS
  _DEFAULT_STROKE: '#000000',
  _DEFAULT_FILL: '#FFFFFF'

};

},{}],37:[function(_dereq_,module,exports){
/**
 * @module Structure
 * @submodule Structure
 * @for p5
 * @requires constants
 */

'use strict';

_dereq_('./shim');

// Core needs the PVariables object
var constants = _dereq_('./constants');

/**
 * This is the p5 instance constructor.
 *
 * A p5 instance holds all the properties and methods related to
 * a p5 sketch.  It expects an incoming sketch closure and it can also
 * take an optional node parameter for attaching the generated p5 canvas
 * to a node.  The sketch closure takes the newly created p5 instance as
 * its sole argument and may optionally set preload(), setup(), and/or
 * draw() properties on it for running a sketch.
 *
 * A p5 sketch can run in "global" or "instance" mode:
 * "global"   - all properties and methods are attached to the window
 * "instance" - all properties and methods are bound to this p5 object
 *
 * @param  {Function}    sketch a closure that can set optional preload(),
 *                              setup(), and/or draw() properties on the
 *                              given p5 instance
 * @param  {HTMLElement|boolean} node element to attach canvas to, if a
 *                                    boolean is passed in use it as sync
 * @param  {boolean}     [sync] start synchronously (optional)
 * @return {p5}                 a p5 instance
 */
var p5 = function(sketch, node, sync) {

  if (arguments.length === 2 && typeof node === 'boolean') {
    sync = node;
    node = undefined;
  }

  //////////////////////////////////////////////
  // PUBLIC p5 PROPERTIES AND METHODS
  //////////////////////////////////////////////


  /**
   * Called directly before setup(), the preload() function is used to handle
   * asynchronous loading of external files. If a preload function is
   * defined, setup() will wait until any load calls within have finished.
   * Nothing besides load calls should be inside preload (loadImage,
   * loadJSON, loadFont, loadStrings, etc).
   *
   * @method preload
   * @example
   * <div><code>
   * var img;
   * var c;
   * function preload() {  // preload() runs once
   *   img = loadImage('assets/laDefense.jpg');
   * }
   *
   * function setup() {  // setup() waits until preload() is done
   *   img.loadPixels();
   *   // get color of middle pixel
   *   c = img.get(img.width/2, img.height/2);
   * }
   *
   * function draw() {
   *   background(c);
   *   image(img, 25, 25, 50, 50);
   * }
   * </code></div>
   */

  /**
   * The setup() function is called once when the program starts. It's used to
   * define initial environment properties such as screen size and background
   * color and to load media such as images and fonts as the program starts.
   * There can only be one setup() function for each program and it shouldn't
   * be called again after its initial execution.
   * <br><br>
   * Note: Variables declared within setup() are not accessible within other
   * functions, including draw().
   *
   * @method setup
   * @example
   * <div><code>
   * var a = 0;
   *
   * function setup() {
   *   background(0);
   *   noStroke();
   *   fill(102);
   * }
   *
   * function draw() {
   *   rect(a++%width, 10, 2, 80);
   * }
   * </code></div>
   */

  /**
   * Called directly after setup(), the draw() function continuously executes
   * the lines of code contained inside its block until the program is stopped
   * or noLoop() is called. Note if noLoop() is called in setup(), draw() will
   * still be executed once before stopping. draw() is called automatically and
   * should never be called explicitly.
   * <br><br>
   * It should always be controlled with noLoop(), redraw() and loop(). After
   * noLoop() stops the code in draw() from executing, redraw() causes the
   * code inside draw() to execute once, and loop() will cause the code
   * inside draw() to resume executing continuously.
   * <br><br>
   * The number of times draw() executes in each second may be controlled with
   * the frameRate() function.
   * <br><br>
   * There can only be one draw() function for each sketch, and draw() must
   * exist if you want the code to run continuously, or to process events such
   * as mousePressed(). Sometimes, you might have an empty call to draw() in
   * your program, as shown in the above example.
   * <br><br>
   * It is important to note that the drawing coordinate system will be reset
   * at the beginning of each draw() call. If any transformations are performed
   * within draw() (ex: scale, rotate, translate, their effects will be
   * undone at the beginning of draw(), so transformations will not accumulate
   * over time. On the other hand, styling applied (ex: fill, stroke, etc) will
   * remain in effect.
   *
   * @method draw
   * @example
   * <div><code>
   * var yPos = 0;
   * function setup() {  // setup() runs once
   *   frameRate(30);
   * }
   * function draw() {  // draw() loops forever, until stopped
   *   background(204);
   *   yPos = yPos - 1;
   *   if (yPos < 0) {
   *     yPos = height;
   *   }
   *   line(0, yPos, width, yPos);
   * }
   * </code></div>
   */


  //////////////////////////////////////////////
  // PRIVATE p5 PROPERTIES AND METHODS
  //////////////////////////////////////////////

  this._setupDone = false;
  // for handling hidpi
  this._pixelDensity = Math.ceil(window.devicePixelRatio) || 1;
  this._userNode = node;
  this._curElement = null;
  this._elements = [];
  this._requestAnimId = 0;
  this._preloadCount = 0;
  this._isGlobal = false;
  this._loop = true;
  this._styles = [];
  this._defaultCanvasSize = {
    width: 100,
    height: 100
  };
  this._events = { // keep track of user-events for unregistering later
    'mousemove': null,
    'mousedown': null,
    'mouseup': null,
    'dragend': null,
    'dragover': null,
    'click': null,
    'mouseover': null,
    'mouseout': null,
    'keydown': null,
    'keyup': null,
    'keypress': null,
    'touchstart': null,
    'touchmove': null,
    'touchend': null,
    'resize': null,
    'blur': null
  };

  this._events.wheel = null;
  this._loadingScreenId = 'p5_loading';

  if (window.DeviceOrientationEvent) {
    this._events.deviceorientation = null;
  }
  if (window.DeviceMotionEvent && !window._isNodeWebkit) {
    this._events.devicemotion = null;
  }

  this._start = function () {
    // Find node if id given
    if (this._userNode) {
      if (typeof this._userNode === 'string') {
        this._userNode = document.getElementById(this._userNode);
      }
    }

    // Always create a default canvas.
    // Later on if the user calls createCanvas, this default one
    // will be replaced
    this.createCanvas(
      this._defaultCanvasSize.width,
      this._defaultCanvasSize.height,
      'p2d',
      true
    );

    var userPreload = this.preload || window.preload; // look for "preload"
    if (userPreload) {

      // Setup loading screen
      // Set loading scfeen into dom if not present
      // Otherwise displays and removes user provided loading screen
      var loadingScreen = document.getElementById(this._loadingScreenId);
      if(!loadingScreen){
        loadingScreen = document.createElement('div');
        loadingScreen.innerHTML = 'Loading...';
        loadingScreen.style.position = 'absolute';
        loadingScreen.id = this._loadingScreenId;
        var node = this._userNode || document.body;
        node.appendChild(loadingScreen);
      }
      // var methods = this._preloadMethods;
      for (var method in this._preloadMethods){
        // default to p5 if no object defined
        this._preloadMethods[method] = this._preloadMethods[method] || p5;
        var obj = this._preloadMethods[method];
        //it's p5, check if it's global or instance
        if (obj === p5.prototype || obj === p5){
          obj = this._isGlobal ? window : this;
        }
        this._registeredPreloadMethods[method] = obj[method];
        obj[method] = this._wrapPreload(obj, method);
      }

      userPreload();
      this._runIfPreloadsAreDone();
    } else {
      this._setup();
      this._runFrames();
      this._draw();
    }
  }.bind(this);

  this._runIfPreloadsAreDone = function(){
    var context = this._isGlobal ? window : this;
    if (context._preloadCount === 0) {
      var loadingScreen = document.getElementById(context._loadingScreenId);
      if (loadingScreen) {
        loadingScreen.parentNode.removeChild(loadingScreen);
      }
      context._setup();
      context._runFrames();
      context._draw();
    }
  };

  this._decrementPreload = function(){
    var context = this._isGlobal ? window : this;
    context._setProperty('_preloadCount', context._preloadCount - 1);
    context._runIfPreloadsAreDone();
  };

  this._wrapPreload = function(obj, fnName){
    return function(){
      //increment counter
      this._incrementPreload();
      //call original function
      var args = new Array(arguments.length);
      for (var i = 0; i < args.length; ++i) {
        args[i] = arguments[i];
      }
      args.push(this._decrementPreload.bind(this));
      return this._registeredPreloadMethods[fnName].apply(obj, args);
    }.bind(this);
  };

  this._incrementPreload = function(){
    var context = this._isGlobal ? window : this;
    context._setProperty('_preloadCount', context._preloadCount + 1);
  };

  this._setup = function() {

    // return preload functions to their normal vals if switched by preload
    var context = this._isGlobal ? window : this;
    if (typeof context.preload === 'function') {
      for (var f in this._preloadMethods) {
        context[f] = this._preloadMethods[f][f];
        if (context[f] && this) {
          context[f] = context[f].bind(this);
        }
      }
    }

    // Short-circuit on this, in case someone used the library in "global"
    // mode earlier
    if (typeof context.setup === 'function') {
      context.setup();
    }

    // unhide any hidden canvases that were created
    var canvases = document.getElementsByTagName('canvas');
    for (var i = 0; i < canvases.length; i++) {
      var k = canvases[i];
      if (k.dataset.hidden === 'true') {
        k.style.visibility = '';
        delete(k.dataset.hidden);
      }
    }
    this._setupDone = true;

  }.bind(this);

  this._draw = function () {
    var now = window.performance.now();
    var time_since_last = now - this._lastFrameTime;
    var target_time_between_frames = 1000 / this._targetFrameRate;

    // only draw if we really need to; don't overextend the browser.
    // draw if we're within 5ms of when our next frame should paint
    // (this will prevent us from giving up opportunities to draw
    // again when it's really about time for us to do so). fixes an
    // issue where the frameRate is too low if our refresh loop isn't
    // in sync with the browser. note that we have to draw once even
    // if looping is off, so we bypass the time delay if that
    // is the case.
    var epsilon = 5;
    if (!this._loop ||
        time_since_last >= target_time_between_frames - epsilon) {

      //mandatory update values(matrixs and stack)
      this.resetMatrix();
      if(this._renderer.isP3D){
        this._renderer._update();
      }

      this._setProperty('frameCount', this.frameCount + 1);
      this.redraw();
      this._updateMouseCoords();
      this._updateTouchCoords();
      this._frameRate = 1000.0/(now - this._lastFrameTime);
      this._lastFrameTime = now;
    }

    // get notified the next time the browser gives us
    // an opportunity to draw.
    if (this._loop) {
      this._requestAnimId = window.requestAnimationFrame(this._draw);
    }
  }.bind(this);

  this._runFrames = function() {
    if (this._updateInterval) {
      clearInterval(this._updateInterval);
    }
  }.bind(this);

  this._setProperty = function(prop, value) {
    this[prop] = value;
    if (this._isGlobal) {
      window[prop] = value;
    }
  }.bind(this);

  /**
   * Removes the entire p5 sketch. This will remove the canvas and any
   * elements created by p5.js. It will also stop the draw loop and unbind
   * any properties or methods from the window global scope. It will
   * leave a variable p5 in case you wanted to create a new p5 sketch.
   * If you like, you can set p5 = null to erase it.
   * @method remove
   * @example
   * <div class='norender'><code>
   * function draw() {
   *   ellipse(50, 50, 10, 10);
   * }
   *
   * function mousePressed() {
   *   remove(); // remove whole sketch on mouse press
   * }
   * </code></div>
   */
  this.remove = function() {
    if (this._curElement) {

      // stop draw
      this._loop = false;
      if (this._requestAnimId) {
        window.cancelAnimationFrame(this._requestAnimId);
      }

      // unregister events sketch-wide
      for (var ev in this._events) {
        window.removeEventListener(ev, this._events[ev]);
      }

      // remove DOM elements created by p5, and listeners
      for (var i=0; i<this._elements.length; i++) {
        var e = this._elements[i];
        if (e.elt.parentNode) {
          e.elt.parentNode.removeChild(e.elt);
        }
        for (var elt_ev in e._events) {
          e.elt.removeEventListener(elt_ev, e._events[elt_ev]);
        }
      }

      // call any registered remove functions
      var self = this;
      this._registeredMethods.remove.forEach(function (f) {
        if (typeof(f) !== 'undefined') {
          f.call(self);
        }
      });

      // remove window bound properties and methods
      if (this._isGlobal) {
        for (var p in p5.prototype) {
          try {
            delete window[p];
          } catch (x) {
            window[p] = undefined;
          }
        }
        for (var p2 in this) {
          if (this.hasOwnProperty(p2)) {
            try {
              delete window[p2];
            } catch (x) {
              window[p2] = undefined;
            }
          }
        }
      }
    }
    // window.p5 = undefined;
  }.bind(this);

  // call any registered init functions
  this._registeredMethods.init.forEach(function (f) {
    if (typeof(f) !== 'undefined') {
      f.call(this);
    }
  }, this);

  var friendlyBindGlobal = this._createFriendlyGlobalFunctionBinder();

  // If the user has created a global setup or draw function,
  // assume "global" mode and make everything global (i.e. on the window)
  if (!sketch) {
    this._isGlobal = true;
    // Loop through methods on the prototype and attach them to the window
    for (var p in p5.prototype) {
      if(typeof p5.prototype[p] === 'function') {
        var ev = p.substring(2);
        if (!this._events.hasOwnProperty(ev)) {
          friendlyBindGlobal(p, p5.prototype[p].bind(this));
        }
      } else {
        friendlyBindGlobal(p, p5.prototype[p]);
      }
    }
    // Attach its properties to the window
    for (var p2 in this) {
      if (this.hasOwnProperty(p2)) {
        friendlyBindGlobal(p2, this[p2]);
      }
    }

  } else {
    // Else, the user has passed in a sketch closure that may set
    // user-provided 'setup', 'draw', etc. properties on this instance of p5
    sketch(this);
  }

  // Bind events to window (not using container div bc key events don't work)

  for (var e in this._events) {
    var f = this['_on'+e];
    if (f) {
      var m = f.bind(this);
      window.addEventListener(e, m);
      this._events[e] = m;
    }
  }

  var focusHandler = function() {
    this._setProperty('focused', true);
  }.bind(this);
  var blurHandler = function() {
    this._setProperty('focused', false);
  }.bind(this);
  window.addEventListener('focus', focusHandler);
  window.addEventListener('blur', blurHandler);
  this.registerMethod('remove', function() {
    window.removeEventListener('focus', focusHandler);
    window.removeEventListener('blur', blurHandler);
  });

  if (sync) {
    this._start();
  } else {
    if (document.readyState === 'complete') {
      this._start();
    } else {
      window.addEventListener('load', this._start.bind(this), false);
    }
  }
};

// attach constants to p5 prototype
for (var k in constants) {
  p5.prototype[k] = constants[k];
}

// functions that cause preload to wait
// more can be added by using registerPreloadMethod(func)
p5.prototype._preloadMethods = {
  loadJSON: p5.prototype,
  loadImage: p5.prototype,
  loadStrings: p5.prototype,
  loadXML: p5.prototype,
  loadShape: p5.prototype,
  loadTable: p5.prototype,
  loadFont: p5.prototype,
  loadModel: p5.prototype
};

p5.prototype._registeredMethods = { init: [], pre: [], post: [], remove: [] };

p5.prototype._registeredPreloadMethods = {};

p5.prototype.registerPreloadMethod = function(fnString, obj) {
  // obj = obj || p5.prototype;
  if (!p5.prototype._preloadMethods.hasOwnProperty(fnString)) {
    p5.prototype._preloadMethods[fnString] = obj;
  }
};

p5.prototype.registerMethod = function(name, m) {
  if (!p5.prototype._registeredMethods.hasOwnProperty(name)) {
    p5.prototype._registeredMethods[name] = [];
  }
  p5.prototype._registeredMethods[name].push(m);
};

p5.prototype._createFriendlyGlobalFunctionBinder = function(options) {
  options = options || {};

  var globalObject = options.globalObject || window;
  var log = options.log || console.log.bind(console);
  var propsToForciblyOverwrite = {
    // p5.print actually always overwrites an existing global function,
    // albeit one that is very unlikely to be used:
    //
    //   https://developer.mozilla.org/en-US/docs/Web/API/Window/print
    'print': true
  };

  return function(prop, value) {
    if (typeof(IS_MINIFIED) === 'undefined' &&
        typeof(value) === 'function' &&
        !(prop in p5.prototype._preloadMethods)) {
      try {
        // Because p5 has so many common function names, it's likely
        // that users may accidentally overwrite global p5 functions with
        // their own variables. Let's allow this but log a warning to
        // help users who may be doing this unintentionally.
        //
        // For more information, see:
        //
        //   https://github.com/processing/p5.js/issues/1317

        if (prop in globalObject && !(prop in propsToForciblyOverwrite)) {
          throw new Error('global "' + prop + '" already exists');
        }

        // It's possible that this might throw an error because there
        // are a lot of edge-cases in which `Object.defineProperty` might
        // not succeed; since this functionality is only intended to
        // help beginners anyways, we'll just catch such an exception
        // if it occurs, and fall back to legacy behavior.
        Object.defineProperty(globalObject, prop, {
          configurable: true,
          enumerable: true,
          get: function() {
            return value;
          },
          set: function(newValue) {
            Object.defineProperty(globalObject, prop, {
              configurable: true,
              enumerable: true,
              value: newValue,
              writable: true
            });
            log(
              'You just changed the value of "' + prop + '", which was ' +
              'a p5 function. This could cause problems later if you\'re ' +
              'not careful.'
            );
          }
        });
      } catch (e) {
        log(
          'p5 had problems creating the global function "' + prop + '", ' +
          'possibly because your code is already using that name as ' +
          'a variable. You may want to rename your variable to something ' +
          'else.'
        );
        globalObject[prop] = value;
      }
    } else {
      globalObject[prop] = value;
    }
  };
};

module.exports = p5;

},{"./constants":36,"./shim":46}],38:[function(_dereq_,module,exports){
/**
 * @module Shape
 * @submodule Curves
 * @for p5
 * @requires core
 */

'use strict';

var p5 = _dereq_('./core');

_dereq_('./error_helpers');

var bezierDetail = 20;
var curveDetail = 20;

/**
 * Draws a cubic Bezier curve on the screen. These curves are defined by a
 * series of anchor and control points. The first two parameters specify
 * the first anchor point and the last two parameters specify the other
 * anchor point, which become the first and last points on the curve. The
 * middle parameters specify the two control points which define the shape
 * of the curve. Approximately speaking, control points "pull" the curve
 * towards them.<br /><br />Bezier curves were developed by French
 * automotive engineer Pierre Bezier, and are commonly used in computer
 * graphics to define gently sloping curves. See also curve().
 *
 * @method bezier
 * @param  {Number} x1 x-coordinate for the first anchor point
 * @param  {Number} y1 y-coordinate for the first anchor point
 * @param  {Number} x2 x-coordinate for the first control point
 * @param  {Number} y2 y-coordinate for the first control point
 * @param  {Number} x3 x-coordinate for the second control point
 * @param  {Number} y3 y-coordinate for the second control point
 * @param  {Number} x4 x-coordinate for the second anchor point
 * @param  {Number} y4 y-coordinate for the second anchor point
 * @return {Object}    the p5 object
 * @example
 * <div>
 * <code>
 * noFill();
 * stroke(255, 102, 0);
 * line(85, 20, 10, 10);
 * line(90, 90, 15, 80);
 * stroke(0, 0, 0);
 * bezier(85, 20, 10, 10, 90, 90, 15, 80);
 * </code>
 * </div>
 */
/**
 * @method bezier
 * @param  {Number} z1 z-coordinate for the first anchor point
 * @param  {Number} z2 z-coordinate for the first control point
 * @param  {Number} z3 z-coordinate for the first anchor point
 * @param  {Number} z4 z-coordinate for the first control point
 * @return {p5.Renderer3D}   [description]
 * @example
 * <div>
 * <code>
 *background(0, 0, 0);
 *noFill();
 *stroke(255);
 *bezier(250,250,0, 100,100,0, 100,0,0, 0,100,0);
 * </code>
 * </div>
*/
p5.prototype.bezier = function() {
   var args = new Array(arguments.length);
   for (var i = 0; i < args.length; ++i) {
     args[i] = arguments[i];
   }
   if(this._renderer.isP3D){
     this._validateParameters(
       'bezier',
       args,
       ['Number', 'Number', 'Number',
       'Number', 'Number', 'Number',
       'Number', 'Number', 'Number',
       'Number', 'Number', 'Number'
       ]
     );
   } else{
     this._validateParameters(
         'bezier',
         args,
         [ 'Number', 'Number', 'Number', 'Number',
           'Number', 'Number', 'Number', 'Number' ]
       );
   }
   if (!this._renderer._doStroke) {
     return this;
   }
   if (this._renderer.isP3D){
     args.push(bezierDetail);//adding value of bezier detail to the args array
     this._renderer.bezier(args);
   } else{
     this._renderer.bezier(args[0],args[1],
       args[2],args[3],
       args[4],args[5],
       args[6],args[7]);
   }

   return this;
 };

/**
 * Sets the resolution at which Beziers display.
 *
 * The default value is 20.
 *
 * @param {Number} detail resolution of the curves
 * @return {Object} the p5 object
 * @example
 * <div>
 * <code>
 * background(204);
 * bezierDetail(50);
 * bezier(85, 20, 10, 10, 90, 90, 15, 80);
 * </code>
 * </div>
 */
p5.prototype.bezierDetail = function(d) {
  bezierDetail = d;
  return this;
};

/**
 * Evaluates the Bezier at position t for points a, b, c, d.
 * The parameters a and d are the first and last points
 * on the curve, and b and c are the control points.
 * The final parameter t varies between 0 and 1.
 * This can be done once with the x coordinates and a second time
 * with the y coordinates to get the location of a bezier curve at t.
 *
 * @method bezierPoint
 * @param {Number} a coordinate of first point on the curve
 * @param {Number} b coordinate of first control point
 * @param {Number} c coordinate of second control point
 * @param {Number} d coordinate of second point on the curve
 * @param {Number} t value between 0 and 1
 * @return {Number} the value of the Bezier at position t
 * @example
 * <div>
 * <code>
 * noFill();
 * x1 = 85, x2 = 10, x3 = 90, x4 = 15;
 * y1 = 20, y2 = 10, y3 = 90, y4 = 80;
 * bezier(x1, y1, x2, y2, x3, y3, x4, y4);
 * fill(255);
 * steps = 10;
 * for (i = 0; i <= steps; i++) {
 *   t = i / steps;
 *   x = bezierPoint(x1, x2, x3, x4, t);
 *   y = bezierPoint(y1, y2, y3, y4, t);
 *   ellipse(x, y, 5, 5);
 * }
 * </code>
 * </div>
 */
p5.prototype.bezierPoint = function(a, b, c, d, t) {
  var adjustedT = 1-t;
  return Math.pow(adjustedT,3)*a +
   3*(Math.pow(adjustedT,2))*t*b +
   3*adjustedT*Math.pow(t,2)*c +
   Math.pow(t,3)*d;
};

/**
 * Evaluates the tangent to the Bezier at position t for points a, b, c, d.
 * The parameters a and d are the first and last points
 * on the curve, and b and c are the control points.
 * The final parameter t varies between 0 and 1.
 *
 * @method bezierTangent
 * @param {Number} a coordinate of first point on the curve
 * @param {Number} b coordinate of first control point
 * @param {Number} c coordinate of second control point
 * @param {Number} d coordinate of second point on the curve
 * @param {Number} t value between 0 and 1
 * @return {Number} the tangent at position t
 * @example
 * <div>
 * <code>
 * noFill();
 * bezier(85, 20, 10, 10, 90, 90, 15, 80);
 * steps = 6;
 * fill(255);
 * for (i = 0; i <= steps; i++) {
 *   t = i / steps;
 *   // Get the location of the point
 *   x = bezierPoint(85, 10, 90, 15, t);
 *   y = bezierPoint(20, 10, 90, 80, t);
 *   // Get the tangent points
 *   tx = bezierTangent(85, 10, 90, 15, t);
 *   ty = bezierTangent(20, 10, 90, 80, t);
 *   // Calculate an angle from the tangent points
 *   a = atan2(ty, tx);
 *   a += PI;
 *   stroke(255, 102, 0);
 *   line(x, y, cos(a)*30 + x, sin(a)*30 + y);
 *   // The following line of code makes a line
 *   // inverse of the above line
 *   //line(x, y, cos(a)*-30 + x, sin(a)*-30 + y);
 *   stroke(0);
 *   ellipse(x, y, 5, 5);
 * }
 * </code>
 * </div>
 *
 * <div>
 * <code>
 * noFill();
 * bezier(85, 20, 10, 10, 90, 90, 15, 80);
 * stroke(255, 102, 0);
 * steps = 16;
 * for (i = 0; i <= steps; i++) {
 *   t = i / steps;
 *   x = bezierPoint(85, 10, 90, 15, t);
 *   y = bezierPoint(20, 10, 90, 80, t);
 *   tx = bezierTangent(85, 10, 90, 15, t);
 *   ty = bezierTangent(20, 10, 90, 80, t);
 *   a = atan2(ty, tx);
 *   a -= HALF_PI;
 *   line(x, y, cos(a)*8 + x, sin(a)*8 + y);
 * }
 * </code>
 * </div>
 */
p5.prototype.bezierTangent = function(a, b, c, d, t) {
  var adjustedT = 1-t;
  return 3*d*Math.pow(t,2) -
   3*c*Math.pow(t,2) +
   6*c*adjustedT*t -
   6*b*adjustedT*t +
   3*b*Math.pow(adjustedT,2) -
   3*a*Math.pow(adjustedT,2);
};

/**
 * Draws a curved line on the screen between two points, given as the
 * middle four parameters. The first two parameters are a control point, as
 * if the curve came from this point even though it's not drawn. The last
 * two parameters similarly describe the other control point. <br /><br />
 * Longer curves can be created by putting a series of curve() functions
 * together or using curveVertex(). An additional function called
 * curveTightness() provides control for the visual quality of the curve.
 * The curve() function is an implementation of Catmull-Rom splines.
 *
 * @method curve
 * @param  {Number} x1 x-coordinate for the beginning control point
 * @param  {Number} y1 y-coordinate for the beginning control point
 * @param  {Number} x2 x-coordinate for the first point
 * @param  {Number} y2 y-coordinate for the first point
 * @param  {Number} x3 x-coordinate for the second point
 * @param  {Number} y3 y-coordinate for the second point
 * @param  {Number} x4 x-coordinate for the ending control point
 * @param  {Number} y4 y-coordinate for the ending control point
 * @return {Object}    the p5 object
 * @example
 * <div>
 * <code>
 * noFill();
 * stroke(255, 102, 0);
 * curve(5, 26, 5, 26, 73, 24, 73, 61);
 * stroke(0);
 * curve(5, 26, 73, 24, 73, 61, 15, 65);
 * stroke(255, 102, 0);
 * curve(73, 24, 73, 61, 15, 65, 15, 65);
 * </code>
 * </div>
 * <div>
 * <code>
 * // Define the curve points as JavaScript objects
 * p1 = {x: 5, y: 26}, p2 = {x: 73, y: 24}
 * p3 = {x: 73, y: 61}, p4 = {x: 15, y: 65}
 * noFill();
 * stroke(255, 102, 0);
 * curve(p1.x, p1.y, p1.x, p1.y, p2.x, p2.y, p3.x, p3.y)
 * stroke(0);
 * curve(p1.x, p1.y, p2.x, p2.y, p3.x, p3.y, p4.x, p4.y)
 * stroke(255, 102, 0);
 * curve(p2.x, p2.y, p3.x, p3.y, p4.x, p4.y, p4.x, p4.y)
 * </code>
 * </div>
 */
/**
 * @method curve
 * @param  {Number} z1 z-coordinate for the beginning control point
 * @param  {Number} z2 z-coordinate for the first point
 * @param  {Number} z3 z-coordinate for the second point
 * @param  {Number} z4 z-coordinate for the ending control point
 * @return {Object}    the p5 object
 * @example
 * <div>
 * <code>
 * noFill();
 * stroke(255, 102, 0);
 * curve(5,26,0, 5,26,0, 73,24,0, 73,61,0);
 * stroke(0);
 * curve(5,26,0, 73,24,0, 73,61,0, 15,65,0);
 * stroke(255, 102, 0);
 * curve(73,24,0, 73,61,0, 15,65,0, 15,65,0);
 * </code>
 * </div>
 */
p5.prototype.curve = function() {
     var args = new Array(arguments.length);
     for (var i = 0; i < args.length; ++i) {
       args[i] = arguments[i];
     }
     if(this._renderer.isP3D){
       this._validateParameters(
         'curve',
         args,
         ['Number', 'Number', 'Number',
         'Number', 'Number', 'Number',
         'Number', 'Number', 'Number',
         'Number', 'Number', 'Number'
         ]
       );
     } else{
       this._validateParameters(
         'curve',
         args,
         [ 'Number', 'Number', 'Number', 'Number',
           'Number', 'Number', 'Number', 'Number' ]
       );
     }
     if (!this._renderer._doStroke) {
       return this;
     }
     if (this._renderer.isP3D){
       args.push(curveDetail);
       this._renderer.curve(args);
     } else{
       this._renderer.curve(args[0],args[1],
         args[2],args[3],
         args[4],args[5],
         args[6],args[7]);
     }
     return this;
   };

/**
 * Sets the resolution at which curves display.
 *
 * The default value is 20.
 *
 * @param {Number} resolution of the curves
 * @return {Object} the p5 object
 * @example
 * <div>
 * <code>
 * background(204);
 * curveDetail(20);
 * curve(5, 26, 5, 26, 73, 24, 73, 61);
 * </code>
 * </div>
 */
p5.prototype.curveDetail = function(d) {
  curveDetail = d;
  return this;
};

/**
 * Modifies the quality of forms created with curve() and curveVertex().
 * The parameter tightness determines how the curve fits to the vertex
 * points. The value 0.0 is the default value for tightness (this value
 * defines the curves to be Catmull-Rom splines) and the value 1.0 connects
 * all the points with straight lines. Values within the range -5.0 and 5.0
 * will deform the curves but will leave them recognizable and as values
 * increase in magnitude, they will continue to deform.
 *
 * @method curveTightness
 * @param {Number} amount of deformation from the original vertices
 * @return {Object} the p5 object
 * @example
 * <div>
 * <code>
 * // Move the mouse left and right to see the curve change
 *
 * function setup() {
 *   createCanvas(100, 100);
 *   noFill();
 * }
 *
 * function draw() {
 *   background(204);
 *   var t = map(mouseX, 0, width, -5, 5);
 *   curveTightness(t);
 *   beginShape();
 *   curveVertex(10, 26);
 *   curveVertex(10, 26);
 *   curveVertex(83, 24);
 *   curveVertex(83, 61);
 *   curveVertex(25, 65);
 *   curveVertex(25, 65);
 *   endShape();
 * }
 * </code>
 * </div>
 */
p5.prototype.curveTightness = function (t) {
  this._renderer._curveTightness = t;
};

/**
 * Evaluates the curve at position t for points a, b, c, d.
 * The parameter t varies between 0 and 1, a and d are points
 * on the curve, and b and c are the control points.
 * This can be done once with the x coordinates and a second time
 * with the y coordinates to get the location of a curve at t.
 *
 * @method curvePoint
 * @param {Number} a coordinate of first point on the curve
 * @param {Number} b coordinate of first control point
 * @param {Number} c coordinate of second control point
 * @param {Number} d coordinate of second point on the curve
 * @param {Number} t value between 0 and 1
 * @return {Number} bezier value at position t
 * @example
 * <div>
 * <code>
 * noFill();
 * curve(5, 26, 5, 26, 73, 24, 73, 61);
 * curve(5, 26, 73, 24, 73, 61, 15, 65);
 * fill(255);
 * ellipseMode(CENTER);
 * steps = 6;
 * for (i = 0; i <= steps; i++) {
 *   t = i / steps;
 *   x = curvePoint(5, 5, 73, 73, t);
 *   y = curvePoint(26, 26, 24, 61, t);
 *   ellipse(x, y, 5, 5);
 *   x = curvePoint(5, 73, 73, 15, t);
 *   y = curvePoint(26, 24, 61, 65, t);
 *   ellipse(x, y, 5, 5);
 * }
 * </code>
 * </div>
 */
p5.prototype.curvePoint = function(a, b, c, d, t) {
  var t3 = t*t*t,
    t2 = t*t,
    f1 = -0.5 * t3 + t2 - 0.5 * t,
    f2 = 1.5 * t3 - 2.5 * t2 + 1.0,
    f3 = -1.5 * t3 + 2.0 * t2 + 0.5 * t,
    f4 = 0.5 * t3 - 0.5 * t2;
  return a*f1 + b*f2 + c*f3 + d*f4;
};

/**
 * Evaluates the tangent to the curve at position t for points a, b, c, d.
 * The parameter t varies between 0 and 1, a and d are points on the curve,
 * and b and c are the control points.
 *
 * @method curveTangent
 * @param {Number} a coordinate of first point on the curve
 * @param {Number} b coordinate of first control point
 * @param {Number} c coordinate of second control point
 * @param {Number} d coordinate of second point on the curve
 * @param {Number} t value between 0 and 1
 * @return {Number} the tangent at position t
 * @example
 * <div>
 * <code>
 * noFill();
 * curve(5, 26, 73, 24, 73, 61, 15, 65);
 * steps = 6;
 * for (i = 0; i <= steps; i++) {
 *   t = i / steps;
 *   x = curvePoint(5, 73, 73, 15, t);
 *   y = curvePoint(26, 24, 61, 65, t);
 *   //ellipse(x, y, 5, 5);
 *   tx = curveTangent(5, 73, 73, 15, t);
 *   ty = curveTangent(26, 24, 61, 65, t);
 *   a = atan2(ty, tx);
 *   a -= PI/2.0;
 *   line(x, y, cos(a)*8 + x, sin(a)*8 + y);
 * }
 * </code>
 * </div>
 */
p5.prototype.curveTangent = function(a, b,c, d, t) {
  var t2 = t*t,
    f1 = (-3*t2)/2 + 2*t - 0.5,
    f2 = (9*t2)/2 - 5*t,
    f3 = (-9*t2)/2 + 4*t + 0.5,
    f4 = (3*t2)/2 - t;
  return a*f1 + b*f2 + c*f3 + d*f4;
};

module.exports = p5;

},{"./core":37,"./error_helpers":40}],39:[function(_dereq_,module,exports){
/**
 * @module Environment
 * @submodule Environment
 * @for p5
 * @requires core
 * @requires constants
 */

'use strict';

var p5 = _dereq_('./core');
var C = _dereq_('./constants');

var standardCursors = [C.ARROW, C.CROSS, C.HAND, C.MOVE, C.TEXT, C.WAIT];

p5.prototype._frameRate = 0;
p5.prototype._lastFrameTime = window.performance.now();
p5.prototype._targetFrameRate = 60;


if (window.console && console.log) {
  /**
   * The print() function writes to the console area of your browser.
   * This function is often helpful for looking at the data a program is
   * producing. This function creates a new line of text for each call to
   * the function. Individual elements can be
   * separated with quotes ("") and joined with the addition operator (+).
   * <br><br>
   * While print() is similar to console.log(), it does not directly map to
   * it in order to simulate easier to understand behavior than
   * console.log(). Due to this, it is slower. For fastest results, use
   * console.log().
   *
   * @method print
   * @param {Any} contents any combination of Number, String, Object, Boolean,
   *                       Array to print
   * @example
   * <div><code class='norender'>
   * var x = 10;
   * print("The value of x is " + x);
   * // prints "The value of x is 10"
   * </code></div>
   */
  // Converts passed args into a string and then parses that string to
  // simulate synchronous behavior. This is a hack and is gross.
  // Since this will not work on all objects, particularly circular
  // structures, simply console.log() on error.
  p5.prototype.print = function(args) {
    try {
      var newArgs = JSON.parse(JSON.stringify(args));
      console.log(newArgs);
    } catch(err) {
      console.log(args);
    }
  };
} else {
  p5.prototype.print = function() {};
}

p5.prototype.println = p5.prototype.print;

/**
 * The system variable frameCount contains the number of frames that have
 * been displayed since the program started. Inside setup() the value is 0,
 * after the first iteration of draw it is 1, etc.
 *
 * @property frameCount
 * @example
 *   <div><code>
 *     function setup() {
 *       frameRate(30);
 *       textSize(20);
 *       textSize(30);
 *       textAlign(CENTER);
 *     }
 *
 *     function draw() {
 *       background(200);
 *       text(frameCount, width/2, height/2);
 *     }
 *   </code></div>
 */
p5.prototype.frameCount = 0;

/**
 * Confirms if the window a p5.js program is in is "focused," meaning that
 * the sketch will accept mouse or keyboard input. This variable is
 * "true" if the window is focused and "false" if not.
 *
 * @property focused
 * @example
 * <div><code>
 * // To demonstrate, put two windows side by side.
 * // Click on the window that the p5 sketch isn't in!
 * function draw() {
 *   background(200);
 *   noStroke();
 *   fill(0, 200, 0);
 *   ellipse(25, 25, 50, 50);
 *
 *   if (!focused) {  // or "if (focused === false)"
 *     stroke(200,0,0);
 *     line(0, 0, 100, 100);
 *     line(100, 0, 0, 100);
 *   }
 * }
 * </code></div>
 */
p5.prototype.focused = (document.hasFocus());

/**
 * Sets the cursor to a predefined symbol or an image, or makes it visible
 * if already hidden. If you are trying to set an image as the cursor, the
 * recommended size is 16x16 or 32x32 pixels. It is not possible to load an
 * image as the cursor if you are exporting your program for the Web, and not
 * all MODES work with all browsers. The values for parameters x and y must
 * be less than the dimensions of the image.
 *
 * @method cursor
 * @param {Number/Constant} type either ARROW, CROSS, HAND, MOVE, TEXT, or
 *                               WAIT, or path for image
 * @param {Number}          [x]  the horizontal active spot of the cursor
 * @param {Number}          [y]  the vertical active spot of the cursor
 * @example
 * <div><code>
 * // Move the mouse left and right across the image
 * // to see the cursor change from a cross to a hand
 * function draw() {
 *   line(width/2, 0, width/2, height);
 *   if (mouseX < 50) {
 *     cursor(CROSS);
 *   } else {
 *     cursor(HAND);
 *   }
 * }
 * </code></div>
 */
p5.prototype.cursor = function(type, x, y) {
  var cursor = 'auto';
  var canvas = this._curElement.elt;
  if (standardCursors.indexOf(type) > -1) {
    // Standard css cursor
    cursor = type;
  } else if (typeof type === 'string') {
    var coords = '';
    if (x && y && (typeof x === 'number' && typeof y === 'number')) {
      // Note that x and y values must be unit-less positive integers < 32
      // https://developer.mozilla.org/en-US/docs/Web/CSS/cursor
      coords = x + ' ' + y;
    }
    if (type.substring(0, 6) !== 'http://') {
      // Image (absolute url)
      cursor = 'url(' + type + ') ' + coords + ', auto';
    } else if (/\.(cur|jpg|jpeg|gif|png|CUR|JPG|JPEG|GIF|PNG)$/.test(type)) {
      // Image file (relative path) - Separated for performance reasons
      cursor = 'url(' + type + ') ' + coords + ', auto';
    } else {
      // Any valid string for the css cursor property
      cursor = type;
    }
  }
  canvas.style.cursor = cursor;
};

/**
 * Specifies the number of frames to be displayed every second. For example,
 * the function call frameRate(30) will attempt to refresh 30 times a second.
 * If the processor is not fast enough to maintain the specified rate, the
 * frame rate will not be achieved. Setting the frame rate within setup() is
 * recommended. The default rate is 60 frames per second. This is the same as
 * setFrameRate(val).
 * <br><br>
 * Calling frameRate() with no arguments returns the current framerate. This
 * is the same as getFrameRate().
 * <br><br>
 * Calling frameRate() with arguments that are not of the type numbers
 * or are non positive also returns current framerate.
 *
 * @method frameRate
 * @param  {Number} [fps] number of frames to be displayed every second
 * @return {Number}       current frameRate
 * @example
 *
 * <div><code>
 * var rectX = 0;
 * var fr = 30; //starting FPS
 * var clr;
 *
 * function setup() {
 *   background(200);
 *   frameRate(fr); // Attempt to refresh at starting FPS
 *   clr = color(255,0,0);
 * }
 *
 * function draw() {
 *   background(200);
 *   rectX = rectX += 1; // Move Rectangle
 *
 *   if (rectX >= width) { // If you go off screen.
 *     if (fr == 30) {
 *       clr = color(0,0,255);
 *       fr = 10;
 *       frameRate(fr); // make frameRate 10 FPS
 *     } else {
 *       clr = color(255,0,0);
 *       fr = 30;
 *       frameRate(fr); // make frameRate 30 FPS
 *     }
 *     rectX = 0;
 *   }
 *   fill(clr);
 *   rect(rectX, 40, 20,20);
 * }
 * </div></code>
 *
 */
p5.prototype.frameRate = function(fps) {
  if (typeof fps !== 'number' || fps <= 0) {
    return this._frameRate;
  } else {
    this._setProperty('_targetFrameRate', fps);
    this._runFrames();
    return this;
  }
};
/**
 * Returns the current framerate.
 *
 * @return {Number} current frameRate
 */
p5.prototype.getFrameRate = function() {
  return this.frameRate();
};

/**
 * Specifies the number of frames to be displayed every second. For example,
 * the function call frameRate(30) will attempt to refresh 30 times a second.
 * If the processor is not fast enough to maintain the specified rate, the
 * frame rate will not be achieved. Setting the frame rate within setup() is
 * recommended. The default rate is 60 frames per second.
 *
 * Calling frameRate() with no arguments returns the current framerate.
 *
 * @param {Number} [fps] number of frames to be displayed every second
 */
p5.prototype.setFrameRate = function(fps) {
  return this.frameRate(fps);
};

/**
 * Hides the cursor from view.
 *
 * @method noCursor
 * @example
 * <div><code>
 * function setup() {
 *   noCursor();
 * }
 *
 * function draw() {
 *   background(200);
 *   ellipse(mouseX, mouseY, 10, 10);
 * }
 * </code></div>
 */
p5.prototype.noCursor = function() {
  this._curElement.elt.style.cursor = 'none';
};


/**
 * System variable that stores the width of the entire screen display. This
 * is used to run a full-screen program on any display size.
 *
 * @property displayWidth
 * @example
 * <div class="norender"><code>
 * createCanvas(displayWidth, displayHeight);
 * </code></div>
 */
p5.prototype.displayWidth = screen.width;

/**
 * System variable that stores the height of the entire screen display. This
 * is used to run a full-screen program on any display size.
 *
 * @property displayHeight
 * @example
 * <div class="norender"><code>
 * createCanvas(displayWidth, displayHeight);
 * </code></div>
 */
p5.prototype.displayHeight = screen.height;

/**
 * System variable that stores the width of the inner window, it maps to
 * window.innerWidth.
 *
 * @property windowWidth
 * @example
 * <div class="norender"><code>
 * createCanvas(windowWidth, windowHeight);
 * </code></div>
 */
p5.prototype.windowWidth = document.documentElement.clientWidth;
/**
 * System variable that stores the height of the inner window, it maps to
 * window.innerHeight.
 *
 * @property windowHeight
 * @example
 * <div class="norender"><code>
 * createCanvas(windowWidth, windowHeight);
 * </code></div>
 */
p5.prototype.windowHeight = document.documentElement.clientHeight;

/**
 * The windowResized() function is called once every time the browser window
 * is resized. This is a good place to resize the canvas or do any other
 * adjustements to accomodate the new window size.
 *
 * @method windowResized
 * @example
 * <div class="norender"><code>
 * function setup() {
 *   createCanvas(windowWidth, windowHeight);
 * }
 *
 * function draw() {
 *  background(0, 100, 200);
 * }
 *
 * function windowResized() {
 *   resizeCanvas(windowWidth, windowHeight);
 * }
 * </code></div>
 */
p5.prototype._onresize = function(e){
  this._setProperty('windowWidth', window.innerWidth);
  this._setProperty('windowHeight', window.innerHeight);
  var context = this._isGlobal ? window : this;
  var executeDefault;
  if (typeof context.windowResized === 'function') {
    executeDefault = context.windowResized(e);
    if (executeDefault !== undefined && !executeDefault) {
      e.preventDefault();
    }
  }
};

/**
 * System variable that stores the width of the drawing canvas. This value
 * is set by the first parameter of the createCanvas() function.
 * For example, the function call createCanvas(320, 240) sets the width
 * variable to the value 320. The value of width defaults to 100 if
 * createCanvas() is not used in a program.
 *
 * @property width
 */
p5.prototype.width = 0;

/**
 * System variable that stores the height of the drawing canvas. This value
 * is set by the second parameter of the createCanvas() function. For
 * example, the function call createCanvas(320, 240) sets the height
 * variable to the value 240. The value of height defaults to 100 if
 * createCanvas() is not used in a program.
 *
 * @property height
 */
p5.prototype.height = 0;

/**
 * If argument is given, sets the sketch to fullscreen or not based on the
 * value of the argument. If no argument is given, returns the current
 * fullscreen state. Note that due to browser restrictions this can only
 * be called on user input, for example, on mouse press like the example
 * below.
 *
 * @method fullscreen
 * @param  {Boolean} [val] whether the sketch should be in fullscreen mode
 * or not
 * @return {Boolean} current fullscreen state
 * @example
 * <div>
 * <code>
 * // Clicking in the box toggles fullscreen on and off.
 * function setup() {
 *   background(200);
 * }
 * function mousePressed() {
 *   if (mouseX > 0 && mouseX < 100 && mouseY > 0 && mouseY < 100) {
 *     var fs = fullscreen();
 *     fullscreen(!fs);
 *   }
 * }
 * </code>
 * </div>
 */
p5.prototype.fullscreen = function(val) {
  // no arguments, return fullscreen or not
  if (typeof val === 'undefined') {
    return document.fullscreenElement ||
           document.webkitFullscreenElement ||
           document.mozFullScreenElement ||
           document.msFullscreenElement;
  } else { // otherwise set to fullscreen or not
    if (val) {
      launchFullscreen(document.documentElement);
    } else {
      exitFullscreen();
    }
  }
};

/**
 * Sets the pixel scaling for high pixel density displays. By default
 * pixel density is set to match display density, call pixelDensity(1)
 * to turn this off. Calling pixelDensity() with no arguments returns
 * the current pixel density of the sketch.
 *
 *
 * @method pixelDensity
 * @param  {Number} [val] whether or how much the sketch should scale
 * @returns {Number} current pixel density of the sketch
 * @example
 * <div>
 * <code>
 * function setup() {
 *   pixelDensity(1);
 *   createCanvas(100, 100);
 *   background(200);
 *   ellipse(width/2, height/2, 50, 50);
 * }
 * </code>
 * </div>
 * <div>
 * <code>
 * function setup() {
 *   pixelDensity(3.0);
 *   createCanvas(100, 100);
 *   background(200);
 *   ellipse(width/2, height/2, 50, 50);
 * }
 * </code>
 * </div>
 */
p5.prototype.pixelDensity = function(val) {
  if (typeof val === 'number') {
    this._pixelDensity = val;
  } else {
    return this._pixelDensity;
  }
  this.resizeCanvas(this.width, this.height, true);
};

/**
 * Returns the pixel density of the current display the sketch is running on.
 *
 * @method displayDensity
 * @returns {Number} current pixel density of the display
 * @example
 * <div>
 * <code>
 * function setup() {
 *   var density = displayDensity();
 *   pixelDensity(density);
 *   createCanvas(100, 100);
 *   background(200);
 *   ellipse(width/2, height/2, 50, 50);
 * }
 * </code>
 * </div>
 */
p5.prototype.displayDensity = function() {
  return window.devicePixelRatio;
};

function launchFullscreen(element) {
  var enabled = document.fullscreenEnabled ||
                document.webkitFullscreenEnabled ||
                document.mozFullScreenEnabled ||
                document.msFullscreenEnabled;
  if (!enabled) {
    throw new Error('Fullscreen not enabled in this browser.');
  }
  if(element.requestFullscreen) {
    element.requestFullscreen();
  } else if(element.mozRequestFullScreen) {
    element.mozRequestFullScreen();
  } else if(element.webkitRequestFullscreen) {
    element.webkitRequestFullscreen();
  } else if(element.msRequestFullscreen) {
    element.msRequestFullscreen();
  }
}

function exitFullscreen() {
  if(document.exitFullscreen) {
    document.exitFullscreen();
  } else if(document.mozCancelFullScreen) {
    document.mozCancelFullScreen();
  } else if(document.webkitExitFullscreen) {
    document.webkitExitFullscreen();
  } else if (document.msExitFullscreen) {
    document.msExitFullscreen();
  }
}


/**
 * Gets the current URL.
 * @method getURL
 * @return {String} url
 * @example
 * <div>
 * <code>
 * var url;
 * var x = 100;
 *
 * function setup() {
 *   fill(0);
 *   noStroke();
 *   url = getURL();
 * }
 *
 * function draw() {
 *   background(200);
 *   text(url, x, height/2);
 *   x--;
 * }
 * </code>
 * </div>
 */
p5.prototype.getURL = function() {
  return location.href;
};
/**
 * Gets the current URL path as an array.
 * @method getURLPath
 * @return {Array} path components
 * @example
 * <div class='norender'><code>
 * function setup() {
 *   var urlPath = getURLPath();
 *   for (var i=0; i&lt;urlPath.length; i++) {
 *     text(urlPath[i], 10, i*20+20);
 *   }
 * }
 * </code></div>
 */
p5.prototype.getURLPath = function() {
  return location.pathname.split('/').filter(function(v){return v!=='';});
};
/**
 * Gets the current URL params as an Object.
 * @method getURLParams
 * @return {Object} URL params
 * @example
 * <div class='norender'>
 * <code>
 * // Example: http://p5js.org?year=2014&month=May&day=15
 *
 * function setup() {
 *   var params = getURLParams();
 *   text(params.day, 10, 20);
 *   text(params.month, 10, 40);
 *   text(params.year, 10, 60);
 * }
 * </code>
 * </div>
 */
p5.prototype.getURLParams = function() {
  var re = /[?&]([^&=]+)(?:[&=])([^&=]+)/gim;
  var m;
  var v={};
  while ((m = re.exec(location.search)) != null) {
    if (m.index === re.lastIndex) {
      re.lastIndex++;
    }
    v[m[1]]=m[2];
  }
  return v;
};

module.exports = p5;

},{"./constants":36,"./core":37}],40:[function(_dereq_,module,exports){
/**
 * @for p5
 * @requires core
 */

'use strict';

var p5 = _dereq_('./core');
var doFriendlyWelcome = false; // TEMP until we get it all working LM

// -- Borrowed from jQuery 1.11.3 --
var class2type = {};
var toString = class2type.toString;
var names = ['Boolean', 'Number', 'String', 'Function',
             'Array', 'Date', 'RegExp', 'Object', 'Error'];
for (var n=0; n<names.length; n++) {
  class2type[ '[object ' + names[n] + ']' ] = names[n].toLowerCase();
}
var getType = function( obj ) {
  if ( obj == null ) {
    return obj + '';
  }
  return typeof obj === 'object' || typeof obj === 'function' ?
    class2type[ toString.call(obj) ] || 'object' :
    typeof obj;
};
var isArray = Array.isArray || function( obj ) {
  return getType(obj) === 'array';
};
var isNumeric =function( obj ) {
  // parseFloat NaNs numeric-cast false positives (null|true|false|"")
  // ...but misinterprets leading-number strings, particularly hex literals
  // subtraction forces infinities to NaN
  // adding 1 corrects loss of precision from parseFloat (#15100)
  return !isArray( obj ) && (obj - parseFloat( obj ) + 1) >= 0;
};
// -- End borrow --

/**
 * Checks the definition type against the argument type
 * If any of these passes (in order), it matches:
 *
 * - p5.* definitions are checked with instanceof
 * - Booleans are let through (everything is truthy or falsey)
 * - Lowercase of the definition is checked against the js type
 * - Number types are checked to see if they are numerically castable
 */
var numberTypes = ['Number', 'Integer', 'Number/Constant'];
function typeMatches(defType, argType, arg) {
  if(defType.match(/^p5\./)) {
    var parts = defType.split('.');
    return arg instanceof p5[parts[1]];
  }
  return defType === 'Boolean' || // Anything is truthy, cover in Debug Guide
    (defType.toLowerCase() === argType) ||
    (numberTypes.indexOf(defType) > -1 && isNumeric(arg));
}

/**
 * Prints out a fancy, colorful message to the console log
 *
 * @param  {String}               message the words to be said
 * @param  {String}               func    the name of the function to link
 * @param  {Integer/Color String} color   CSS color string or error type
 *
 * @return console logs
 */
// Wrong number of params, undefined param, wrong type
var PARAM_COUNT = 0;
var EMPTY_VAR = 1;
var WRONG_TYPE = 2;
var FILE_LOAD = 3;
// p5.js blue, p5.js orange, auto dark green; fallback p5.js darkened magenta
// See testColors below for all the color codes and names
var typeColors = ['#2D7BB6', '#EE9900', '#4DB200', '#C83C00'];
function report(message, func, color) {
  if(doFriendlyWelcome){
    friendlyWelcome();
    doFriendlyWelcome =false;
  }
  if ('undefined' === getType(color)) {
    color   = '#B40033'; // dark magenta
  } else if (getType(color) === 'number') { // Type to color
    color = typeColors[color];
  }
  // LM TEMP commenting this out until we get the whole system working
  // if (func.substring(0,4) === 'load'){
  //   console.log(
  //     '%c> p5.js says: '+message+'%c'+
  //     '[https://github.com/processing/p5.js/wiki/Local-server]',
  //     'background-color:' + color + ';color:#FFF;',
  //     'background-color:transparent;color:' + color +';',
  //     'background-color:' + color + ';color:#FFF;',
  //     'background-color:transparent;color:' + color +';'
  //   );
  // }
  // else{
  //   console.log(
  //     '%c> p5.js says: '+message+'%c [http://p5js.org/reference/#p5/'+func+
  //     ']', 'background-color:' + color + ';color:#FFF;',
  //     'background-color:transparent;color:' + color +';'
  //   );
  // }
}

/**
 * Validate all the parameters of a function for number and type
 * NOTE THIS FUNCTION IS TEMPORARILY DISABLED UNTIL FURTHER WORK
 * AND UPDATES ARE IMPLEMENTED. -LMCCART
 *
 * @param  {String} func  name of function we're checking
 * @param  {Array}  args  pass of the JS default arguments array
 * @param  {Array}  types List of types accepted ['Number', 'String, ...] OR
 *                        a list of lists for each format: [
 *                          ['String', 'Number', 'Number'],
 *                          ['String', 'Number', 'Number', 'Number', 'Number'
 *                        ]
 *
 * @return console logs
 */
p5.prototype._validateParameters = function(func, args, types) {
  if (!isArray(types[0])) {
    types = [types];
  }
  // Check number of parameters
  // Example: "You wrote ellipse(X,X,X). ellipse was expecting 4
  //          parameters. Try ellipse(X,X,X,X)."
  var diff = Math.abs(args.length-types[0].length);
  var message, tindex = 0;
  for (var i=1, len=types.length; i<len; i++) {
    var d = Math.abs(args.length-types[i].length);
    if (d <= diff) {
      tindex = i;
      diff = d;
    }
  }
  var symbol = 'X'; // Parameter placeholder
  if(diff > 0) {
    message = 'You wrote ' + func + '(';
    // Concat an appropriate number of placeholders for call
    if (args.length > 0) {
      message += symbol + Array(args.length).join(',' + symbol);
    }
    message += '). ' + func + ' was expecting ' + types[tindex].length +
      ' parameters. Try ' + func + '(';
    // Concat an appropriate number of placeholders for definition
    if (types[tindex].length > 0) {
      message += symbol + Array(types[tindex].length).join(',' + symbol);
    }
    message += ').';
    // If multiple definitions
    if (types.length > 1) {
      message += ' ' + func + ' takes different numbers of parameters ' +
        'depending on what you want to do. Click this link to learn more: ';
    }
    report(message, func, PARAM_COUNT);
  }
  // Type checking
  // Example: "It looks like ellipse received an empty variable in spot #2."
  // Example: "ellipse was expecting a number for parameter #1,
  //           received "foo" instead."
  for (var format=0; format<types.length; format++) {
    for (var p=0; p < types[format].length && p < args.length; p++) {
      var defType = types[format][p];
      var argType = getType(args[p]);
      if ('undefined' === argType || null === argType) {
        report('It looks like ' + func +
          ' received an empty variable in spot #' + (p+1) +
          '. If not intentional, this is often a problem with scope: ' +
          '[link to scope].', func, EMPTY_VAR);
      } else if (defType !== '*' && !typeMatches(defType, argType, args[p])) {
        message = func + ' was expecting a ' + defType.toLowerCase() +
          ' for parameter #' + (p+1) + ', received ';
        // Wrap strings in quotes
        message += 'string' === argType ? '"' + args[p] + '"' : args[p];
        message += ' instead.';
        // If multiple definitions
        if (types.length > 1) {
          message += ' ' + func + ' takes different numbers of parameters ' +
            'depending on what you want to do. ' +
            'Click this link to learn more:';
        }
        report(message, func, WRONG_TYPE);
      }
    }
  }
};
/*
 * NOTE THIS FUNCTION IS TEMPORARILY DISABLED UNTIL FURTHER WORK
 * AND UPDATES ARE IMPLEMENTED. -LMCCART
 */
p5.prototype._validateParameters = function() {
  return true;
};

var errorCases = {
  '0': {
    fileType: 'image',
    method: 'loadImage',
    message: ' hosting the image online,'
  },
  '1': {
    fileType: 'XML file',
    method: 'loadXML'
  },
  '2': {
    fileType: 'table file',
    method: 'loadTable'
  },
  '3': {
    fileType: 'text file',
    method: 'loadStrings'
  }
};
p5._friendlyFileLoadError = function (errorType, filePath) {
  var errorInfo = errorCases[ errorType ];
  var message = 'It looks like there was a problem' +
  ' loading your ' + errorInfo.fileType + '.' +
  ' Try checking if the file path%c [' + filePath + '] %cis correct,' +
  (errorInfo.message || '') + ' or running a local server.';
  report(message, errorInfo.method, FILE_LOAD);
};

function friendlyWelcome() {
  // p5.js brand - magenta: #ED225D
  var astrixBgColor = 'transparent';
  var astrixTxtColor = '#ED225D';
  var welcomeBgColor = '#ED225D';
  var welcomeTextColor = 'white';
  console.log(
  '%c    _ \n'+
  ' /\\| |/\\ \n'+
  ' \\ ` \' /  \n'+
  ' / , . \\  \n'+
  ' \\/|_|\\/ '+
  '\n\n%c> p5.js says: Welcome! '+
  'This is your friendly debugger. ' +
  'To turn me off switch to using “p5.min.js”.',
  'background-color:'+astrixBgColor+';color:' + astrixTxtColor +';',
  'background-color:'+welcomeBgColor+';color:' + welcomeTextColor +';'
  );
}

/**
 * Prints out all the colors in the color pallete with white text.
 * For color blindness testing.
 */
/* function testColors() {
  var str = 'A box of biscuits, a box of mixed biscuits and a biscuit mixer';
  report(str, 'println', '#ED225D'); // p5.js magenta
  report(str, 'println', '#2D7BB6'); // p5.js blue
  report(str, 'println', '#EE9900'); // p5.js orange
  report(str, 'println', '#A67F59'); // p5.js light brown
  report(str, 'println', '#704F21'); // p5.js gold
  report(str, 'println', '#1CC581'); // auto cyan
  report(str, 'println', '#FF6625'); // auto orange
  report(str, 'println', '#79EB22'); // auto green
  report(str, 'println', '#B40033'); // p5.js darkened magenta
  report(str, 'println', '#084B7F'); // p5.js darkened blue
  report(str, 'println', '#945F00'); // p5.js darkened orange
  report(str, 'println', '#6B441D'); // p5.js darkened brown
  report(str, 'println', '#2E1B00'); // p5.js darkened gold
  report(str, 'println', '#008851'); // auto dark cyan
  report(str, 'println', '#C83C00'); // auto dark orange
  report(str, 'println', '#4DB200'); // auto dark green
} */

// This is a lazily-defined list of p5 symbols that may be
// misused by beginners at top-level code, outside of setup/draw. We'd like
// to detect these errors and help the user by suggesting they move them
// into setup/draw.
//
// For more details, see https://github.com/processing/p5.js/issues/1121.
var misusedAtTopLevelCode = null;
var FAQ_URL = 'https://github.com/processing/p5.js/wiki/' +
              'Frequently-Asked-Questions' +
              '#why-cant-i-assign-variables-using-p5-functions-and-' +
              'variables-before-setup';

function defineMisusedAtTopLevelCode() {
  var uniqueNamesFound = {};

  var getSymbols = function(obj) {
    return Object.getOwnPropertyNames(obj).filter(function(name) {
      if (name[0] === '_') {
        return false;
      }
      if (name in uniqueNamesFound) {
        return false;
      }

      uniqueNamesFound[name] = true;

      return true;
    }).map(function(name) {
      var type;

      if (typeof(obj[name]) === 'function') {
        type = 'function';
      } else if (name === name.toUpperCase()) {
        type = 'constant';
      } else {
        type = 'variable';
      }

      return {name: name, type: type};
    });
  };

  misusedAtTopLevelCode = [].concat(
    getSymbols(p5.prototype),
    // At present, p5 only adds its constants to p5.prototype during
    // construction, which may not have happened at the time a
    // ReferenceError is thrown, so we'll manually add them to our list.
    getSymbols(_dereq_('./constants'))
  );

  // This will ultimately ensure that we report the most specific error
  // possible to the user, e.g. advising them about HALF_PI instead of PI
  // when their code misuses the former.
  misusedAtTopLevelCode.sort(function(a, b) {
    return b.name.length - a.name.length;
  });
}

function helpForMisusedAtTopLevelCode(e, log) {
  if (!log) {
    log = console.log.bind(console);
  }

  if (!misusedAtTopLevelCode) {
    defineMisusedAtTopLevelCode();
  }

  // If we find that we're logging lots of false positives, we can
  // uncomment the following code to avoid displaying anything if the
  // user's code isn't likely to be using p5's global mode. (Note that
  // setup/draw are more likely to be defined due to JS function hoisting.)
  //
  //if (!('setup' in window || 'draw' in window)) {
  //  return;
  //}

  misusedAtTopLevelCode.some(function(symbol) {
    // Note that while just checking for the occurrence of the
    // symbol name in the error message could result in false positives,
    // a more rigorous test is difficult because different browsers
    // log different messages, and the format of those messages may
    // change over time.
    //
    // For example, if the user uses 'PI' in their code, it may result
    // in any one of the following messages:
    //
    //   * 'PI' is undefined                           (Microsoft Edge)
    //   * ReferenceError: PI is undefined             (Firefox)
    //   * Uncaught ReferenceError: PI is not defined  (Chrome)

    if (e.message && e.message.indexOf(symbol.name) !== -1) {
      log('%cDid you just try to use p5.js\'s ' + symbol.name +
          (symbol.type === 'function' ? '() ' : ' ') + symbol.type +
          '? If so, you may want to ' +
          'move it into your sketch\'s setup() function.\n\n' +
          'For more details, see: ' + FAQ_URL,
          'color: #B40033' /* Dark magenta */);
      return true;
    }
  });
}

// Exposing this primarily for unit testing.
p5.prototype._helpForMisusedAtTopLevelCode = helpForMisusedAtTopLevelCode;

if (document.readyState !== 'complete') {
  window.addEventListener('error', helpForMisusedAtTopLevelCode, false);

  // Our job is only to catch ReferenceErrors that are thrown when
  // global (non-instance mode) p5 APIs are used at the top-level
  // scope of a file, so we'll unbind our error listener now to make
  // sure we don't log false positives later.
  window.addEventListener('load', function() {
    window.removeEventListener('error', helpForMisusedAtTopLevelCode, false);
  });
}

module.exports = p5;

},{"./constants":36,"./core":37}],41:[function(_dereq_,module,exports){
/**
 * @module DOM
 * @submodule DOM
 * @for p5.Element
 */

var p5 = _dereq_('./core');

/**
 * Base class for all elements added to a sketch, including canvas,
 * graphics buffers, and other HTML elements. Methods in blue are
 * included in the core functionality, methods in brown are added
 * with the <a href="http://p5js.org/libraries/">p5.dom library</a>.
 * It is not called directly, but p5.Element
 * objects are created by calling createCanvas, createGraphics,
 * or in the p5.dom library, createDiv, createImg, createInput, etc.
 *
 * @class p5.Element
 * @constructor
 * @param {String} elt DOM node that is wrapped
 * @param {Object} [pInst] pointer to p5 instance
 */
p5.Element = function(elt, pInst) {
  /**
   * Underlying HTML element. All normal HTML methods can be called on this.
   *
   * @property elt
   */
  this.elt = elt;
  this._pInst = pInst;
  this._events = {};
  this.width = this.elt.offsetWidth;
  this.height = this.elt.offsetHeight;
};

/**
 *
 * Attaches the element to the parent specified. A way of setting
 * the container for the element. Accepts either a string ID, DOM
 * node, or p5.Element. If no arguments given, parent node is returned.
 * For more ways to position the canvas, see the
 * <a href='https://github.com/processing/p5.js/wiki/Positioning-your-canvas'>
 * positioning the canvas</a> wiki page.
 *
 * @method parent
 * @param  {String|Object} parent the ID, DOM node, or p5.Element
 *                         of desired parent element
 * @return {p5.Element}
 * @example
 * <div class="norender"><code>
 * // in the html file:
 * &lt;div id="myContainer">&lt;/div>
 * // in the js file:
 * var cnv = createCanvas(100, 100);
 * cnv.parent("myContainer");
 * </code></div>
 * <div class='norender'><code>
 * var div0 = createDiv('this is the parent');
 * var div1 = createDiv('this is the child');
 * div1.parent(div0); // use p5.Element
 * </code></div>
 * <div class='norender'><code>
 * var div0 = createDiv('this is the parent');
 * div0.id('apples');
 * var div1 = createDiv('this is the child');
 * div1.parent('apples'); // use id
 * </code></div>
 * <div class='norender'><code>
 * var elt = document.getElementById('myParentDiv');
 * var div1 = createDiv('this is the child');
 * div1.parent(elt); // use element from page
 * </code></div>
 */
p5.Element.prototype.parent = function(p) {
  if (arguments.length === 0){
    return this.elt.parentNode;
  } else {
    if (typeof p === 'string') {
      if (p[0] === '#') {
        p = p.substring(1);
      }
      p = document.getElementById(p);
    } else if (p instanceof p5.Element) {
      p = p.elt;
    }
    p.appendChild(this.elt);
    return this;
  }
};

/**
 *
 * Sets the ID of the element
 *
 * @method id
 * @param  {String} id ID of the element
 * @return {p5.Element}
 */
p5.Element.prototype.id = function(id) {
  if (arguments.length === 0) {
    return this.elt.id;
  } else {
    this.elt.id = id;
    this.width = this.elt.offsetWidth;
    this.height = this.elt.offsetHeight;
    return this;
  }
};

/**
 *
 * Adds given class to the element
 *
 * @method class
 * @param  {String} class class to add
 * @return {p5.Element}
 */
p5.Element.prototype.class = function(c) {
  if (arguments.length === 0) {
    return this.elt.className;
  } else {
    this.elt.className = c;
    return this;
  }
};

/**
 * The .mousePressed() function is called once after every time a
 * mouse button is pressed over the element. This can be used to
 * attach element specific event listeners.
 *
 * @method mousePressed
 * @param  {Function} fxn function to be fired when mouse is
 *                    pressed over the element.
 * @return {p5.Element}
 * @example
 * <div class='norender'><code>
 * var cnv;
 * var d;
 * var g;
 * function setup() {
 *   cnv = createCanvas(100, 100);
 *   cnv.mousePressed(changeGray); // attach listener for
 *                                 // canvas click only
 *   d = 10;
 *   g = 100;
 * }
 *
 * function draw() {
 *   background(g);
 *   ellipse(width/2, height/2, d, d);
 * }
 *
 * // this function fires with any click anywhere
 * function mousePressed() {
 *   d = d + 10;
 * }
 *
 * // this function fires only when cnv is clicked
 * function changeGray() {
 *   g = random(0, 255);
 * }
 * </code></div>
 *
 */
p5.Element.prototype.mousePressed = function (fxn) {
  attachListener('mousedown', fxn, this);
  attachListener('touchstart', fxn, this);
  return this;
};

/**
 * The .mouseWheel() function is called once after every time a
 * mouse wheel is scrolled over the element. This can be used to
 * attach element specific event listeners.<br><br>
 * The event.wheelDelta or event.detail property returns negative values if
 * the mouse wheel if rotated up or away from the user and positive in the
 * other direction. On OS X with "natural" scrolling enabled, the values are
 * opposite.
 *
 * @method mouseWheel
 * @param  {Function} fxn function to be fired when mouse wheel is
 *                    scrolled over the element.
 * @return {p5.Element}
 * @example
 * <div class='norender'><code>
 * var cnv;
 * var d;
 * var g;
 * function setup() {
 *   cnv = createCanvas(100, 100);
 *   cnv.mouseWheel(changeSize); // attach listener for
 *                               // activity on canvas only
 *   d = 10;
 *   g = 100;
 * }
 *
 * function draw() {
 *   background(g);
 *   ellipse(width/2, height/2, d, d);
 * }
 *
 * // this function fires with mousewheel movement
 * // anywhere on screen
 * function mouseWheel() {
 *   g = g + 10;
 * }
 *
 * // this function fires with mousewheel movement
 * // over canvas only
 * function changeSize() {
 *   if (event.wheelDelta > 0) {
 *     d = d + 10;
 *   } else {
 *     d = d - 10;
 *   }
 * }
 * </code></div>
 *
 */
p5.Element.prototype.mouseWheel = function (fxn) {
  attachListener('wheel', fxn, this);
  return this;
};

/**
 * The .mouseReleased() function is called once after every time a
 * mouse button is released over the element. This can be used to
 * attach element specific event listeners.
 *
 * @method mouseReleased
 * @param  {Function} fxn function to be fired when mouse is
 *                    released over the element.
 * @return {p5.Element}
 * @example
 * <div class='norender'><code>
 * var cnv;
 * var d;
 * var g;
 * function setup() {
 *   cnv = createCanvas(100, 100);
 *   cnv.mouseReleased(changeGray); // attach listener for
 *                                  // activity on canvas only
 *   d = 10;
 *   g = 100;
 * }
 *
 * function draw() {
 *   background(g);
 *   ellipse(width/2, height/2, d, d);
 * }
 *
 * // this function fires after the mouse has been
 * // released
 * function mouseReleased() {
 *   d = d + 10;
 * }
 *
 * // this function fires after the mouse has been
 * // released while on canvas
 * function changeGray() {
 *   g = random(0, 255);
 * }
 * </code></div>
 *
 */
p5.Element.prototype.mouseReleased = function (fxn) {
  attachListener('mouseup', fxn, this);
  attachListener('touchend', fxn, this);
  return this;
};


/**
 * The .mouseClicked() function is called once after a mouse button is
 * pressed and released over the element. This can be used to
 * attach element specific event listeners.
 *
 * @method mouseClicked
 * @param  {Function} fxn function to be fired when mouse is
 *                    clicked over the element.
 * @return {p5.Element}
 * @example
 * var cnv;
 * var d;
 * var g;
 * function setup() {
 *   cnv = createCanvas(100, 100);
 *   cnv.mouseClicked(changeGray); // attach listener for
 *                                 // activity on canvas only
 *   d = 10;
 *   g = 100;
 * }
 *
 * function draw() {
 *   background(g);
 *   ellipse(width/2, height/2, d, d);
 * }
 *
 * // this function fires after the mouse has been
 * // clicked anywhere
 * function mouseClicked() {
 *   d = d + 10;
 * }
 *
 * // this function fires after the mouse has been
 * // clicked on canvas
 * function changeGray() {
 *   g = random(0, 255);
 * }
 * </code></div>
 *
 */
p5.Element.prototype.mouseClicked = function (fxn) {
  attachListener('click', fxn, this);
  return this;
};

/**
 * The .mouseMoved() function is called once every time a
 * mouse moves over the element. This can be used to attach an
 * element specific event listener.
 *
 * @method mouseMoved
 * @param  {Function} fxn function to be fired when mouse is
 *                    moved over the element.
 * @return {p5.Element}
 * @example
 * <div class='norender'><code>
 * var cnv;
 * var d = 30;
 * var g;
 * function setup() {
 *   cnv = createCanvas(100, 100);
 *   cnv.mouseMoved(changeSize); // attach listener for
 *                               // activity on canvas only
 *   d = 10;
 *   g = 100;
 * }
 *
 * function draw() {
 *   background(g);
 *   fill(200);
 *   ellipse(width/2, height/2, d, d);
 * }
 *
 * // this function fires when mouse moves anywhere on
 * // page
 * function mouseMoved() {
 *   g = g + 5;
 *   if (g > 255) {
 *     g = 0;
 *   }
 * }
 *
 * // this function fires when mouse moves over canvas
 * function changeSize() {
 *   d = d + 2;
 *   if (d > 100) {
 *     d = 0;
 *   }
 * }
 * </code></div>
 *
 */
p5.Element.prototype.mouseMoved = function (fxn) {
  attachListener('mousemove', fxn, this);
  attachListener('touchmove', fxn, this);
  return this;
};

/**
 * The .mouseOver() function is called once after every time a
 * mouse moves onto the element. This can be used to attach an
 * element specific event listener.
 *
 * @method mouseOver
 * @param  {Function} fxn function to be fired when mouse is
 *                    moved over the element.
 * @return {p5.Element}
 * @example
 * <div class='norender'><code>
 * var cnv;
 * var d;
 * var g;
 * function setup() {
 *   cnv = createCanvas(100, 100);
 *   cnv.mouseOver(changeGray);
 *   d = 10;
 * }
 *
 * function draw() {
 *   ellipse(width/2, height/2, d, d);
 * }
 *
 * function changeGray() {
 *   d = d + 10;
 *   if (d > 100) {
 *     d = 0;
 *   }
 * }
 * </code></div>
 *
 */
p5.Element.prototype.mouseOver = function (fxn) {
  attachListener('mouseover', fxn, this);
  return this;
};


/**
 * The .changed() function is called when the value of an
 * element is changed.
 * This can be used to attach an element specific event listener.
 *
 * @method changed
 * @param  {Function} fxn function to be fired when the value of an
 * element changes.
 * @return {p5.Element}
 * @example
 * <div><code>
 * var sel;
 *
 * function setup() {
 *   textAlign(CENTER);
 *   background(200);
 *   sel = createSelect();
 *   sel.position(10, 10);
 *   sel.option('pear');
 *   sel.option('kiwi');
 *   sel.option('grape');
 *   sel.changed(mySelectEvent);
 * }
 *
 * function mySelectEvent() {
 *   var item = sel.value();
 *   background(200);
 *   text("it's a "+item+"!", 50, 50);
 * }
 * </code></div>
 * <div><code>
 * var checkbox;
 * var cnv;
 *
 * function setup() {
 *   checkbox = createCheckbox(" fill");
 *   checkbox.changed(changeFill);
 *   cnv = createCanvas(100, 100);
 *   cnv.position(0, 30);
 *   noFill();
 * }
 *
 * function draw() {
 *   background(200);
 *   ellipse(50, 50, 50, 50);
 * }
 *
 * function changeFill() {
 *   if (checkbox.checked()) {
 *     fill(0);
 *   } else {
 *     noFill();
 *   }
 * }
 * </code></div>
 */
p5.Element.prototype.changed = function (fxn) {
  attachListener('change', fxn, this);
  return this;
};

/**
 * The .input() function is called when any user input is
 * detected with an element. The input event is often used
 * to detect keystrokes in a input element, or changes on a
 * slider element. This can be used to attach an element specific
 * event listener.
 *
 * @method input
 * @param  {Function} fxn function to be fired on user input.
 * @return {p5.Element}
 * @example
 * <div class='norender'><code>
 * // Open your console to see the output
 * function setup() {
 *   var inp = createInput('');
 *   inp.input(myInputEvent);
 * }
 *
 * function myInputEvent() {
 *   console.log('you are typing: ', this.value());
 * }
 * </code></div>
 *
 */
p5.Element.prototype.input = function (fxn) {
  attachListener('input', fxn, this);
  return this;
};

/**
 * The .mouseOut() function is called once after every time a
 * mouse moves off the element. This can be used to attach an
 * element specific event listener.
 *
 * @method mouseOut
 * @param  {Function} fxn function to be fired when mouse is
 *                    moved off the element.
 * @return {p5.Element}
 * @example
 * <div class='norender'><code>
 * var cnv;
 * var d;
 * var g;
 * function setup() {
 *   cnv = createCanvas(100, 100);
 *   cnv.mouseOut(changeGray);
 *   d = 10;
 * }
 *
 * function draw() {
 *   ellipse(width/2, height/2, d, d);
 * }
 *
 * function changeGray() {
 *   d = d + 10;
 *   if (d > 100) {
 *     d = 0;
 *   }
 * }
 * </code></div>
 *
 */
p5.Element.prototype.mouseOut = function (fxn) {
  attachListener('mouseout', fxn, this);
  return this;
};

/**
 * The .touchStarted() function is called once after every time a touch is
 * registered. This can be used to attach element specific event listeners.
 *
 * @method touchStarted
 * @param  {Function} fxn function to be fired when touch is
 *                    started over the element.
 * @return {p5.Element}
 * @example
 * <div class='norender'><code>
 * var cnv;
 * var d;
 * var g;
 * function setup() {
 *   cnv = createCanvas(100, 100);
 *   cnv.touchStarted(changeGray); // attach listener for
 *                                 // canvas click only
 *   d = 10;
 *   g = 100;
 * }
 *
 * function draw() {
 *   background(g);
 *   ellipse(width/2, height/2, d, d);
 * }
 *
 * // this function fires with any touch anywhere
 * function touchStarted() {
 *   d = d + 10;
 * }
 *
 * // this function fires only when cnv is clicked
 * function changeGray() {
 *   g = random(0, 255);
 * }
 * </code></div>
 *
 */
p5.Element.prototype.touchStarted = function (fxn) {
  attachListener('touchstart', fxn, this);
  attachListener('mousedown', fxn, this);
  return this;
};

/**
 * The .touchMoved() function is called once after every time a touch move is
 * registered. This can be used to attach element specific event listeners.
 *
 * @method touchMoved
 * @param  {Function} fxn function to be fired when touch is moved
 *                    over the element.
 * @return {p5.Element}
 * @example
 * <div class='norender'><code>
 * var cnv;
 * var g;
 * function setup() {
 *   cnv = createCanvas(100, 100);
 *   cnv.touchMoved(changeGray); // attach listener for
 *                               // canvas click only
 *   g = 100;
 * }
 *
 * function draw() {
 *   background(g);
 * }
 *
 * // this function fires only when cnv is clicked
 * function changeGray() {
 *   g = random(0, 255);
 * }
 * </code></div>
 *
 */
p5.Element.prototype.touchMoved = function (fxn) {
  attachListener('touchmove', fxn, this);
  attachListener('mousemove', fxn, this);
  return this;
};

/**
 * The .touchEnded() function is called once after every time a touch is
 * registered. This can be used to attach element specific event listeners.
 *
 * @method touchEnded
 * @param  {Function} fxn function to be fired when touch is
 *                    ended over the element.
 * @return {p5.Element}
 * @example
 * <div class='norender'><code>
 * var cnv;
 * var d;
 * var g;
 * function setup() {
 *   cnv = createCanvas(100, 100);
 *   cnv.touchEnded(changeGray);   // attach listener for
 *                                 // canvas click only
 *   d = 10;
 *   g = 100;
 * }
 *
 * function draw() {
 *   background(g);
 *   ellipse(width/2, height/2, d, d);
 * }
 *
 * // this function fires with any touch anywhere
 * function touchEnded() {
 *   d = d + 10;
 * }
 *
 * // this function fires only when cnv is clicked
 * function changeGray() {
 *   g = random(0, 255);
 * }
 * </code></div>
 *
 */
p5.Element.prototype.touchEnded = function (fxn) {
  attachListener('touchend', fxn, this);
  attachListener('mouseup', fxn, this);
  return this;
};



/**
 * The .dragOver() function is called once after every time a
 * file is dragged over the element. This can be used to attach an
 * element specific event listener.
 *
 * @method dragOver
 * @param  {Function} fxn function to be fired when mouse is
 *                    dragged over the element.
 * @return {p5.Element}
 */
p5.Element.prototype.dragOver = function (fxn) {
  attachListener('dragover', fxn, this);
  return this;
};

/**
 * The .dragLeave() function is called once after every time a
 * dragged file leaves the element area. This can be used to attach an
 * element specific event listener.
 *
 * @method dragLeave
 * @param  {Function} fxn function to be fired when mouse is
 *                    dragged over the element.
 * @return {p5.Element}
 */
p5.Element.prototype.dragLeave = function (fxn) {
  attachListener('dragleave', fxn, this);
  return this;
};

/**
 * The .drop() function is called for each file dropped on the element.
 * It requires a callback that is passed a p5.File object.  You can
 * optionally pass two callbacks, the first one (required) is triggered
 * for each file dropped when the file is loaded.  The second (optional)
 * is triggered just once when a file (or files) are dropped.
 *
 * @method drop
 * @param  {Function} callback triggered when files are dropped.
 * @param  {Function} callback to receive loaded file.
 * @return {p5.Element}
 * @example
 * <div><code>
 * function setup() {
 *   var c = createCanvas(100, 100);
 *   background(200);
 *   textAlign(CENTER);
 *   text('drop image', width/2, height/2);
 *   c.drop(gotFile);
 * }
 *
 * function gotFile(file) {
 *   var img = createImg(file.data).hide();
 *   // Draw the image onto the canvas
 *   image(img, 0, 0, width, height);
 * }
 * </code></div>
 */
p5.Element.prototype.drop = function (callback, fxn) {
  // Make a file loader callback and trigger user's callback
  function makeLoader(theFile) {
    // Making a p5.File object
    var p5file = new p5.File(theFile);
    return function(e) {
      p5file.data = e.target.result;
      callback(p5file);
    };
  }

  // Is the file stuff supported?
  if (window.File && window.FileReader && window.FileList && window.Blob) {

    // If you want to be able to drop you've got to turn off
    // a lot of default behavior
    attachListener('dragover',function(evt) {
      evt.stopPropagation();
      evt.preventDefault();
    },this);

    // If this is a drag area we need to turn off the default behavior
    attachListener('dragleave',function(evt) {
      evt.stopPropagation();
      evt.preventDefault();
    },this);

    // If just one argument it's the callback for the files
    if (arguments.length > 1) {
      attachListener('drop', fxn, this);
    }

    // Deal with the files
    attachListener('drop', function(evt) {

      evt.stopPropagation();
      evt.preventDefault();

      // A FileList
      var files = evt.dataTransfer.files;

      // Load each one and trigger the callback
      for (var i = 0; i < files.length; i++) {
        var f = files[i];
        var reader = new FileReader();
        reader.onload = makeLoader(f);


        // Text or data?
        // This should likely be improved
        if (f.type.indexOf('text') > -1) {
          reader.readAsText(f);
        } else {
          reader.readAsDataURL(f);
        }
      }
    }, this);
  } else {
    console.log('The File APIs are not fully supported in this browser.');
  }

  return this;
};




function attachListener(ev, fxn, ctx) {
  // LM removing, not sure why we had this?
  // var _this = ctx;
  // var f = function (e) { fxn(e, _this); };
  var f = fxn.bind(ctx);
  ctx.elt.addEventListener(ev, f, false);
  ctx._events[ev] = f;
}

/**
 * Helper fxn for sharing pixel methods
 *
 */
p5.Element.prototype._setProperty = function (prop, value) {
  this[prop] = value;
};


module.exports = p5.Element;

},{"./core":37}],42:[function(_dereq_,module,exports){
/**
 * @module Rendering
 * @submodule Rendering
 * @for p5
 */

var p5 = _dereq_('./core');
var constants = _dereq_('./constants');

/**
 * Thin wrapper around a renderer, to be used for creating a
 * graphics buffer object. Use this class if you need
 * to draw into an off-screen graphics buffer. The two parameters define the
 * width and height in pixels. The fields and methods for this class are
 * extensive, but mirror the normal drawing API for p5.
 *
 * @class p5.Graphics
 * @constructor
 * @extends p5.Element
 * @param {String} elt DOM node that is wrapped
 * @param {Object} [pInst] pointer to p5 instance
 * @param {Boolean} whether we're using it as main canvas
 */
p5.Graphics = function(w, h, renderer, pInst) {

  var r = renderer || constants.P2D;

  var c = document.createElement('canvas');
  var node = this._userNode || document.body;
  node.appendChild(c);

  p5.Element.call(this, c, pInst, false);
  this._styles = [];
  this.width = w;
  this.height = h;
  this._pixelDensity = pInst._pixelDensity;

  if (r === constants.WEBGL) {
    this._renderer = new p5.RendererGL(c, this, false);
  } else {
    this._renderer = new p5.Renderer2D(c, this, false);
  }

  this._renderer.resize(w, h);
  this._renderer._applyDefaults();

  pInst._elements.push(this);

  // bind methods and props of p5 to the new object
  for (var p in p5.prototype) {
    if (!this[p]) {
      if (typeof p5.prototype[p] === 'function') {
        this[p] = p5.prototype[p].bind(this);
      } else {
        this[p] = p5.prototype[p];
      }
    }
  }

  return this;
};

p5.Graphics.prototype = Object.create(p5.Element.prototype);

module.exports = p5.Graphics;

},{"./constants":36,"./core":37}],43:[function(_dereq_,module,exports){
/**
 * @module Rendering
 * @submodule Rendering
 * @for p5
 */

var p5 = _dereq_('./core');
var constants = _dereq_('../core/constants');

/**
 * Main graphics and rendering context, as well as the base API
 * implementation for p5.js "core". To be used as the superclass for
 * Renderer2D and Renderer3D classes, respecitvely.
 *
 * @class p5.Renderer
 * @constructor
 * @extends p5.Element
 * @param {String} elt DOM node that is wrapped
 * @param {Object} [pInst] pointer to p5 instance
 * @param {Boolean} whether we're using it as main canvas
 */
p5.Renderer = function(elt, pInst, isMainCanvas) {
  p5.Element.call(this, elt, pInst);
  this.canvas = elt;
  this._pInst = pInst;
  if (isMainCanvas) {
    this._isMainCanvas = true;
    // for pixel method sharing with pimage
    this._pInst._setProperty('_curElement', this);
    this._pInst._setProperty('canvas', this.canvas);
    this._pInst._setProperty('width', this.width);
    this._pInst._setProperty('height', this.height);
  } else { // hide if offscreen buffer by default
    this.canvas.style.display = 'none';
    this._styles = []; // non-main elt styles stored in p5.Renderer
  }


  this._textSize = 12;
  this._textLeading = 15;
  this._textFont = 'sans-serif';
  this._textStyle = constants.NORMAL;
  this._textAscent = null;
  this._textDescent = null;


  this._rectMode = constants.CORNER;
  this._ellipseMode = constants.CENTER;
  this._curveTightness = 0;
  this._imageMode = constants.CORNER;

  this._tint = null;
  this._doStroke = true;
  this._doFill = true;
  this._strokeSet = false;
  this._fillSet = false;
  this._colorMode = constants.RGB;
  this._colorMaxes = {
    rgb: [255, 255, 255, 255],
    hsb: [360, 100, 100, 1],
    hsl: [360, 100, 100, 1]
  };

};

p5.Renderer.prototype = Object.create(p5.Element.prototype);




/**
 * Resize our canvas element.
 */
p5.Renderer.prototype.resize = function(w, h) {
  this.width = w;
  this.height = h;
  this.elt.width = w * this._pInst._pixelDensity;
  this.elt.height = h * this._pInst._pixelDensity;
  this.elt.style.width = w +'px';
  this.elt.style.height = h + 'px';
  if (this._isMainCanvas) {
    this._pInst._setProperty('width', this.width);
    this._pInst._setProperty('height', this.height);
  }
};

p5.Renderer.prototype.textLeading = function(l) {

  if (arguments.length && arguments[0]) {

    this._setProperty('_textLeading', l);
    return this;
  }

  return this._textLeading;
};

p5.Renderer.prototype.textSize = function(s) {

  if (arguments.length && arguments[0]) {

    this._setProperty('_textSize', s);
    this._setProperty('_textLeading', s * constants._DEFAULT_LEADMULT);
    return this._applyTextProperties();
  }

  return this._textSize;
};

p5.Renderer.prototype.textStyle = function(s) {

  if (arguments.length && arguments[0]) {

    if (s === constants.NORMAL ||
      s === constants.ITALIC ||
      s === constants.BOLD) {
      this._setProperty('_textStyle', s);
    }

    return this._applyTextProperties();
  }

  return this._textStyle;
};

p5.Renderer.prototype.textAscent = function() {
  if (this._textAscent === null) {
    this._updateTextMetrics();
  }
  return this._textAscent;
};

p5.Renderer.prototype.textDescent = function() {

  if (this._textDescent === null) {
    this._updateTextMetrics();
  }
  return this._textDescent;
};

p5.Renderer.prototype._applyDefaults = function(){
  return this;
};

/**
 * Helper fxn to check font type (system or otf)
 */
p5.Renderer.prototype._isOpenType = function(f) {

  f = f || this._textFont;
  return (typeof f === 'object' && f.font && f.font.supported);
};

p5.Renderer.prototype._updateTextMetrics = function() {

  if (this._isOpenType()) {

    this._setProperty('_textAscent', this._textFont._textAscent());
    this._setProperty('_textDescent', this._textFont._textDescent());
    return this;
  }

  // Adapted from http://stackoverflow.com/a/25355178
  var text = document.createElement('span');
  text.style.fontFamily = this._textFont;
  text.style.fontSize = this._textSize + 'px';
  text.innerHTML = 'ABCjgq|';

  var block = document.createElement('div');
  block.style.display = 'inline-block';
  block.style.width = '1px';
  block.style.height = '0px';

  var container = document.createElement('div');
  container.appendChild(text);
  container.appendChild(block);

  container.style.height = '0px';
  container.style.overflow = 'hidden';
  document.body.appendChild(container);

  block.style.verticalAlign = 'baseline';
  var blockOffset = calculateOffset(block);
  var textOffset = calculateOffset(text);
  var ascent = blockOffset[1] - textOffset[1];

  block.style.verticalAlign = 'bottom';
  blockOffset = calculateOffset(block);
  textOffset = calculateOffset(text);
  var height = blockOffset[1] - textOffset[1];
  var descent = height - ascent;

  document.body.removeChild(container);

  this._setProperty('_textAscent', ascent);
  this._setProperty('_textDescent', descent);

  return this;
};

/**
 * Helper fxn to measure ascent and descent.
 * Adapted from http://stackoverflow.com/a/25355178
 */
function calculateOffset(object) {
  var currentLeft = 0,
    currentTop = 0;
  if (object.offsetParent) {
    do {
      currentLeft += object.offsetLeft;
      currentTop += object.offsetTop;
    } while (object = object.offsetParent);
  } else {
    currentLeft += object.offsetLeft;
    currentTop += object.offsetTop;
  }
  return [currentLeft, currentTop];
}

module.exports = p5.Renderer;

},{"../core/constants":36,"./core":37}],44:[function(_dereq_,module,exports){

var p5 = _dereq_('./core');
var canvas = _dereq_('./canvas');
var constants = _dereq_('./constants');
var filters = _dereq_('../image/filters');

_dereq_('./p5.Renderer');

/**
 * p5.Renderer2D
 * The 2D graphics canvas renderer class.
 * extends p5.Renderer
 */
var styleEmpty = 'rgba(0,0,0,0)';
// var alphaThreshold = 0.00125; // minimum visible

p5.Renderer2D = function(elt, pInst, isMainCanvas){
  p5.Renderer.call(this, elt, pInst, isMainCanvas);
  this.drawingContext = this.canvas.getContext('2d');
  this._pInst._setProperty('drawingContext', this.drawingContext);
  return this;
};

p5.Renderer2D.prototype = Object.create(p5.Renderer.prototype);

p5.Renderer2D.prototype._applyDefaults = function() {
  this.drawingContext.fillStyle = constants._DEFAULT_FILL;
  this.drawingContext.strokeStyle = constants._DEFAULT_STROKE;
  this.drawingContext.lineCap = constants.ROUND;
  this.drawingContext.font = 'normal 12px sans-serif';
};

p5.Renderer2D.prototype.resize = function(w,h) {
  p5.Renderer.prototype.resize.call(this, w,h);
  this.drawingContext.scale(this._pInst._pixelDensity,
                            this._pInst._pixelDensity);
};

//////////////////////////////////////////////
// COLOR | Setting
//////////////////////////////////////////////

p5.Renderer2D.prototype.background = function() {
  this.drawingContext.save();
  this.drawingContext.setTransform(1, 0, 0, 1, 0, 0);
  this.drawingContext.scale(this._pInst._pixelDensity,
                            this._pInst._pixelDensity);

  if (arguments[0] instanceof p5.Image) {
    this._pInst.image(arguments[0], 0, 0, this.width, this.height);
  } else {
    var curFill = this.drawingContext.fillStyle;
    // create background rect
    var color = this._pInst.color.apply(this, arguments);
    var newFill = color.toString();
    this.drawingContext.fillStyle = newFill;
    this.drawingContext.fillRect(0, 0, this.width, this.height);
    // reset fill
    this.drawingContext.fillStyle = curFill;
  }
  this.drawingContext.restore();
};

p5.Renderer2D.prototype.clear = function() {
  this.drawingContext.clearRect(0, 0, this.width, this.height);
};

p5.Renderer2D.prototype.fill = function() {

  var ctx = this.drawingContext;
  var color = this._pInst.color.apply(this, arguments);
  ctx.fillStyle = color.toString();
};

p5.Renderer2D.prototype.stroke = function() {
  var ctx = this.drawingContext;
  var color = this._pInst.color.apply(this, arguments);
  ctx.strokeStyle = color.toString();
};

//////////////////////////////////////////////
// IMAGE | Loading & Displaying
//////////////////////////////////////////////

p5.Renderer2D.prototype.image =
  function (img, sx, sy, sWidth, sHeight, dx, dy, dWidth, dHeight) {
  var cnv;
  try {
    if (this._tint) {
      if (p5.MediaElement && img instanceof p5.MediaElement) {
        img.loadPixels();
      }
      if (img.canvas) {
        cnv = this._getTintedImageCanvas(img);
      }
    }
    if (!cnv) {
      cnv = img.canvas || img.elt;
    }
    this.drawingContext.drawImage(cnv, sx, sy, sWidth, sHeight, dx, dy,
      dWidth, dHeight);
  } catch (e) {
    if (e.name !== 'NS_ERROR_NOT_AVAILABLE') {
      throw e;
    }
  }
};

p5.Renderer2D.prototype._getTintedImageCanvas = function (img) {
  if (!img.canvas) {
    return img;
  }
  var pixels = filters._toPixels(img.canvas);
  var tmpCanvas = document.createElement('canvas');
  tmpCanvas.width = img.canvas.width;
  tmpCanvas.height = img.canvas.height;
  var tmpCtx = tmpCanvas.getContext('2d');
  var id = tmpCtx.createImageData(img.canvas.width, img.canvas.height);
  var newPixels = id.data;
  for (var i = 0; i < pixels.length; i += 4) {
    var r = pixels[i];
    var g = pixels[i + 1];
    var b = pixels[i + 2];
    var a = pixels[i + 3];
    newPixels[i] = r * this._tint[0] / 255;
    newPixels[i + 1] = g * this._tint[1] / 255;
    newPixels[i + 2] = b * this._tint[2] / 255;
    newPixels[i + 3] = a * this._tint[3] / 255;
  }
  tmpCtx.putImageData(id, 0, 0);
  return tmpCanvas;
};


//////////////////////////////////////////////
// IMAGE | Pixels
//////////////////////////////////////////////

p5.Renderer2D.prototype.blendMode = function(mode) {
  this.drawingContext.globalCompositeOperation = mode;
};
p5.Renderer2D.prototype.blend = function() {
  var currBlend = this.drawingContext.globalCompositeOperation;
  var blendMode = arguments[arguments.length - 1];

  var copyArgs = Array.prototype.slice.call(
    arguments,
    0,
    arguments.length - 1
  );

  this.drawingContext.globalCompositeOperation = blendMode;
  if (this._pInst) {
    this._pInst.copy.apply(this._pInst, copyArgs);
  } else {
    this.copy.apply(this, copyArgs);
  }
  this.drawingContext.globalCompositeOperation = currBlend;
};

p5.Renderer2D.prototype.copy = function () {
  var srcImage, sx, sy, sw, sh, dx, dy, dw, dh;
  if (arguments.length === 9) {
    srcImage = arguments[0];
    sx = arguments[1];
    sy = arguments[2];
    sw = arguments[3];
    sh = arguments[4];
    dx = arguments[5];
    dy = arguments[6];
    dw = arguments[7];
    dh = arguments[8];
  } else if (arguments.length === 8) {
    srcImage = this._pInst;
    sx = arguments[0];
    sy = arguments[1];
    sw = arguments[2];
    sh = arguments[3];
    dx = arguments[4];
    dy = arguments[5];
    dw = arguments[6];
    dh = arguments[7];
  } else {
    throw new Error('Signature not supported');
  }
  p5.Renderer2D._copyHelper(srcImage, sx, sy, sw, sh, dx, dy, dw, dh);
};

p5.Renderer2D._copyHelper =
function (srcImage, sx, sy, sw, sh, dx, dy, dw, dh) {
  var s = srcImage.canvas.width / srcImage.width;
  this.drawingContext.drawImage(srcImage.canvas,
    s * sx, s * sy, s * sw, s * sh, dx, dy, dw, dh);
};

p5.Renderer2D.prototype.get = function(x, y, w, h) {
  if (x === undefined && y === undefined &&
      w === undefined && h === undefined){
    x = 0;
    y = 0;
    w = this.width;
    h = this.height;
  } else if (w === undefined && h === undefined) {
    w = 1;
    h = 1;
  }

  // if the section does not overlap the canvas
  if(x + w < 0 || y + h < 0 || x > this.width || y > this.height){
    return [0, 0, 0, 255];
  }

  var ctx = this._pInst || this;

  var pd = ctx._pixelDensity;
  ctx.loadPixels();

  // round down to get integer numbers
  x = Math.floor(x);
  y = Math.floor(y);

  if (w === 1 && h === 1){
    var startPoint = pd*pd*4*(y*this.width+x);
    return [
      ctx.pixels[startPoint],
      ctx.pixels[startPoint+1],
      ctx.pixels[startPoint+2],
      ctx.pixels[startPoint+3]
    ];
  } else {
    var sx = x * pd;
    var sy = y * pd;
    //auto constrain the width and height to
    //dimensions of the source image
    var dw = Math.min(w, ctx.width);
    var dh = Math.min(h, ctx.height);
    var sw = dw * pd;
    var sh = dh * pd;

    var region = new p5.Image(dw, dh);
    region.canvas.getContext('2d').drawImage(this.canvas, sx, sy, sw, sh,
      0, 0, dw, dh);

    return region;
  }
};

p5.Renderer2D.prototype.loadPixels = function () {
  var pd = this._pixelDensity || this._pInst._pixelDensity;
  var w = this.width * pd;
  var h = this.height * pd;
  var imageData = this.drawingContext.getImageData(0, 0, w, h);
  // @todo this should actually set pixels per object, so diff buffers can
  // have diff pixel arrays.
  if (this._pInst) {
    this._pInst._setProperty('imageData', imageData);
    this._pInst._setProperty('pixels', imageData.data);
  } else { // if called by p5.Image
    this._setProperty('imageData', imageData);
    this._setProperty('pixels', imageData.data);
  }
};

p5.Renderer2D.prototype.set = function (x, y, imgOrCol) {
  // round down to get integer numbers
  x = Math.floor(x);
  y = Math.floor(y);
  if (imgOrCol instanceof p5.Image) {
    this.drawingContext.save();
    this.drawingContext.setTransform(1, 0, 0, 1, 0, 0);
    this.drawingContext.scale(this._pInst._pixelDensity,
      this._pInst._pixelDensity);
    this.drawingContext.drawImage(imgOrCol.canvas, x, y);
    this.loadPixels.call(this._pInst);
    this.drawingContext.restore();
  } else {
    var ctx = this._pInst || this;
    var r = 0, g = 0, b = 0, a = 0;
    var idx = 4*((y * ctx._pixelDensity) *
      (this.width * ctx._pixelDensity) + (x * ctx._pixelDensity));
    if (!ctx.imageData) {
      ctx.loadPixels.call(ctx);
    }
    if (typeof imgOrCol === 'number') {
      if (idx < ctx.pixels.length) {
        r = imgOrCol;
        g = imgOrCol;
        b = imgOrCol;
        a = 255;
        //this.updatePixels.call(this);
      }
    }
    else if (imgOrCol instanceof Array) {
      if (imgOrCol.length < 4) {
        throw new Error('pixel array must be of the form [R, G, B, A]');
      }
      if (idx < ctx.pixels.length) {
        r = imgOrCol[0];
        g = imgOrCol[1];
        b = imgOrCol[2];
        a = imgOrCol[3];
        //this.updatePixels.call(this);
      }
    } else if (imgOrCol instanceof p5.Color) {
      if (idx < ctx.pixels.length) {
        r = imgOrCol.levels[0];
        g = imgOrCol.levels[1];
        b = imgOrCol.levels[2];
        a = imgOrCol.levels[3];
        //this.updatePixels.call(this);
      }
    }
    // loop over pixelDensity * pixelDensity
    for (var i = 0; i < ctx._pixelDensity; i++) {
      for (var j = 0; j < ctx._pixelDensity; j++) {
        // loop over
        idx = 4*((y * ctx._pixelDensity + j) * this.width *
          ctx._pixelDensity + (x * ctx._pixelDensity + i));
        ctx.pixels[idx] = r;
        ctx.pixels[idx+1] = g;
        ctx.pixels[idx+2] = b;
        ctx.pixels[idx+3] = a;
      }
    }
  }
};

p5.Renderer2D.prototype.updatePixels = function (x, y, w, h) {
  var pd = this._pixelDensity || this._pInst._pixelDensity;
  if (x === undefined &&
      y === undefined &&
      w === undefined &&
      h === undefined) {
    x = 0;
    y = 0;
    w = this.width;
    h = this.height;
  }
  w *= pd;
  h *= pd;

  if (this._pInst) {
    this.drawingContext.putImageData(this._pInst.imageData, x, y, 0, 0, w, h);
  } else {
    this.drawingContext.putImageData(this.imageData, x, y, 0, 0, w, h);
  }
};

//////////////////////////////////////////////
// SHAPE | 2D Primitives
//////////////////////////////////////////////

/**
 * Generate a cubic Bezier representing an arc on the unit circle of total
 * angle `size` radians, beginning `start` radians above the x-axis. Up to
 * four of these curves are combined to make a full arc.
 *
 * See www.joecridge.me/bezier.pdf for an explanation of the method.
 */
p5.Renderer2D.prototype._acuteArcToBezier =
  function _acuteArcToBezier(start, size) {
  // Evauate constants.
  var alpha = size / 2.0,
    cos_alpha = Math.cos(alpha),
    sin_alpha = Math.sin(alpha),
    cot_alpha = 1.0 / Math.tan(alpha),
    phi = start + alpha,  // This is how far the arc needs to be rotated.
    cos_phi = Math.cos(phi),
    sin_phi = Math.sin(phi),
    lambda = (4.0 - cos_alpha) / 3.0,
    mu = sin_alpha + (cos_alpha - lambda) * cot_alpha;

  // Return rotated waypoints.
  return {
    ax: Math.cos(start),
    ay: Math.sin(start),
    bx: lambda * cos_phi + mu * sin_phi,
    by: lambda * sin_phi - mu * cos_phi,
    cx: lambda * cos_phi - mu * sin_phi,
    cy: lambda * sin_phi + mu * cos_phi,
    dx: Math.cos(start + size),
    dy: Math.sin(start + size)
  };
};

p5.Renderer2D.prototype.arc =
  function(x, y, w, h, start, stop, mode) {
  var ctx = this.drawingContext;
  var vals = canvas.arcModeAdjust(x, y, w, h, this._ellipseMode);
  var rx = vals.w / 2.0;
  var ry = vals.h / 2.0;
  var epsilon = 0.00001;  // Smallest visible angle on displays up to 4K.
  var arcToDraw = 0;
  var curves = [];

  // Create curves
  while(stop - start > epsilon) {
    arcToDraw = Math.min(stop - start, constants.HALF_PI);
    curves.push(this._acuteArcToBezier(start, arcToDraw));
    start += arcToDraw;
  }

  // Fill curves
  if (this._doFill) {
    ctx.beginPath();
    curves.forEach(function (curve, index) {
      if (index === 0) {
        ctx.moveTo(vals.x + curve.ax * rx, vals.y + curve.ay * ry);
      }
      ctx.bezierCurveTo(vals.x + curve.bx * rx, vals.y + curve.by * ry,
                        vals.x + curve.cx * rx, vals.y + curve.cy * ry,
                        vals.x + curve.dx * rx, vals.y + curve.dy * ry);
    });
    if (mode === constants.PIE || mode == null) {
      ctx.lineTo(vals.x, vals.y);
    }
    ctx.closePath();
    ctx.fill();
  }

  // Stroke curves
  if (this._doStroke) {
    ctx.beginPath();
    curves.forEach(function (curve, index) {
      if (index === 0) {
        ctx.moveTo(vals.x + curve.ax * rx, vals.y + curve.ay * ry);
      }
      ctx.bezierCurveTo(vals.x + curve.bx * rx, vals.y + curve.by * ry,
                        vals.x + curve.cx * rx, vals.y + curve.cy * ry,
                        vals.x + curve.dx * rx, vals.y + curve.dy * ry);
    });
    if (mode === constants.PIE) {
      ctx.lineTo(vals.x, vals.y);
      ctx.closePath();
    } else if (mode === constants.CHORD) {
      ctx.closePath();
    }
    ctx.stroke();
  }
  return this;
};

p5.Renderer2D.prototype.ellipse = function(x, y, w, h) {
  var ctx = this.drawingContext;
  var doFill = this._doFill, doStroke = this._doStroke;
  if (doFill && !doStroke) {
    if(ctx.fillStyle === styleEmpty) {
      return this;
    }
  } else if (!doFill && doStroke) {
    if(ctx.strokeStyle === styleEmpty) {
      return this;
    }
  }
  var vals = canvas.modeAdjust(x, y, w, h, this._ellipseMode);
  var kappa = 0.5522847498,
    ox = (vals.w / 2) * kappa, // control point offset horizontal
    oy = (vals.h / 2) * kappa, // control point offset vertical
    xe = vals.x + vals.w,      // x-end
    ye = vals.y + vals.h,      // y-end
    xm = vals.x + vals.w / 2,  // x-middle
    ym = vals.y + vals.h / 2;  // y-middle
  ctx.beginPath();
  ctx.moveTo(vals.x, ym);
  ctx.bezierCurveTo(vals.x, ym - oy, xm - ox, vals.y, xm, vals.y);
  ctx.bezierCurveTo(xm + ox, vals.y, xe, ym - oy, xe, ym);
  ctx.bezierCurveTo(xe, ym + oy, xm + ox, ye, xm, ye);
  ctx.bezierCurveTo(xm - ox, ye, vals.x, ym + oy, vals.x, ym);
  ctx.closePath();
  if (doFill) {
    ctx.fill();
  }
  if (doStroke) {
    ctx.stroke();
  }
};

p5.Renderer2D.prototype.line = function(x1, y1, x2, y2) {
  var ctx = this.drawingContext;
  if (!this._doStroke) {
    return this;
  } else if(ctx.strokeStyle === styleEmpty){
    return this;
  }
  // Translate the line by (0.5, 0.5) to draw it crisp
  if (ctx.lineWidth % 2 === 1) {
    ctx.translate(0.5, 0.5);
  }
  ctx.beginPath();
  ctx.moveTo(x1, y1);
  ctx.lineTo(x2, y2);
  ctx.stroke();
  if (ctx.lineWidth % 2 === 1) {
    ctx.translate(-0.5, -0.5);
  }
  return this;
};

p5.Renderer2D.prototype.point = function(x, y) {
  var ctx = this.drawingContext;
  var s = ctx.strokeStyle;
  var f = ctx.fillStyle;
  if (!this._doStroke) {
    return this;
  } else if(ctx.strokeStyle === styleEmpty){
    return this;
  }
  x = Math.round(x);
  y = Math.round(y);
  ctx.fillStyle = s;
  if (ctx.lineWidth > 1) {
    ctx.beginPath();
    ctx.arc(
      x,
      y,
      ctx.lineWidth / 2,
      0,
      constants.TWO_PI,
      false
    );
    ctx.fill();
  } else {
    ctx.fillRect(x, y, 1, 1);
  }
  ctx.fillStyle = f;
};

p5.Renderer2D.prototype.quad =
  function(x1, y1, x2, y2, x3, y3, x4, y4) {
  var ctx = this.drawingContext;
  var doFill = this._doFill, doStroke = this._doStroke;
  if (doFill && !doStroke) {
    if(ctx.fillStyle === styleEmpty) {
      return this;
    }
  } else if (!doFill && doStroke) {
    if(ctx.strokeStyle === styleEmpty) {
      return this;
    }
  }
  ctx.beginPath();
  ctx.moveTo(x1, y1);
  ctx.lineTo(x2, y2);
  ctx.lineTo(x3, y3);
  ctx.lineTo(x4, y4);
  ctx.closePath();
  if (doFill) {
    ctx.fill();
  }
  if (doStroke) {
    ctx.stroke();
  }
  return this;
};

p5.Renderer2D.prototype.rect = function(args) {
  var x = args[0],
    y = args[1],
    w = args[2],
    h = args[3],
    tl = args[4],
    tr = args[5],
    br = args[6],
    bl = args[7];
  var ctx = this.drawingContext;
  var doFill = this._doFill, doStroke = this._doStroke;
  if (doFill && !doStroke) {
    if(ctx.fillStyle === styleEmpty) {
      return this;
    }
  } else if (!doFill && doStroke) {
    if(ctx.strokeStyle === styleEmpty) {
      return this;
    }
  }
  var vals = canvas.modeAdjust(x, y, w, h, this._rectMode);
  // Translate the line by (0.5, 0.5) to draw a crisp rectangle border
  if (this._doStroke && ctx.lineWidth % 2 === 1) {
    ctx.translate(0.5, 0.5);
  }
  ctx.beginPath();

  if (typeof tl === 'undefined') {
    // No rounded corners
    ctx.rect(vals.x, vals.y, vals.w, vals.h);
  } else {
    // At least one rounded corner
    // Set defaults when not specified
    if (typeof tr === 'undefined') { tr = tl; }
    if (typeof br === 'undefined') { br = tr; }
    if (typeof bl === 'undefined') { bl = br; }

    // Cache and compute several values
    var _x = vals.x;
    var _y = vals.y;
    var _w = vals.w;
    var _h = vals.h;
    var hw = _w / 2;
    var hh = _h / 2;

    // Clip radii
    if (_w < 2 * tl) { tl = hw; }
    if (_h < 2 * tl) { tl = hh; }
    if (_w < 2 * tr) { tr = hw; }
    if (_h < 2 * tr) { tr = hh; }
    if (_w < 2 * br) { br = hw; }
    if (_h < 2 * br) { br = hh; }
    if (_w < 2 * bl) { bl = hw; }
    if (_h < 2 * bl) { bl = hh; }

    // Draw shape
    ctx.beginPath();
    ctx.moveTo(_x + tl, _y);
    ctx.arcTo(_x + _w, _y, _x + _w, _y + _h, tr);
    ctx.arcTo(_x + _w, _y + _h, _x, _y + _h, br);
    ctx.arcTo(_x, _y + _h, _x, _y, bl);
    ctx.arcTo(_x, _y, _x + _w, _y, tl);
    ctx.closePath();
  }
  if (this._doFill) {
    ctx.fill();
  }
  if (this._doStroke) {
    ctx.stroke();
  }
  if (this._doStroke && ctx.lineWidth % 2 === 1) {
    ctx.translate(-0.5, -0.5);
  }
  return this;
};

p5.Renderer2D.prototype.triangle = function(x1, y1, x2, y2, x3, y3) {
  var ctx = this.drawingContext;
  var doFill = this._doFill, doStroke = this._doStroke;
  if (doFill && !doStroke) {
    if(ctx.fillStyle === styleEmpty) {
      return this;
    }
  } else if (!doFill && doStroke) {
    if(ctx.strokeStyle === styleEmpty) {
      return this;
    }
  }
  ctx.beginPath();
  ctx.moveTo(x1, y1);
  ctx.lineTo(x2, y2);
  ctx.lineTo(x3, y3);
  ctx.closePath();
  if (doFill) {
    ctx.fill();
  }
  if (doStroke) {
    ctx.stroke();
  }
};

p5.Renderer2D.prototype.endShape =
function (mode, vertices, isCurve, isBezier,
    isQuadratic, isContour, shapeKind) {
  if (vertices.length === 0) {
    return this;
  }
  if (!this._doStroke && !this._doFill) {
    return this;
  }
  var closeShape = mode === constants.CLOSE;
  var v;
  if (closeShape && !isContour) {
    vertices.push(vertices[0]);
  }
  var i, j;
  var numVerts = vertices.length;
  if (isCurve && (shapeKind === constants.POLYGON || shapeKind === null)) {
    if (numVerts > 3) {
      var b = [], s = 1 - this._curveTightness;
      this.drawingContext.beginPath();
      this.drawingContext.moveTo(vertices[1][0], vertices[1][1]);
      for (i = 1; i + 2 < numVerts; i++) {
        v = vertices[i];
        b[0] = [
          v[0],
          v[1]
        ];
        b[1] = [
          v[0] + (s * vertices[i + 1][0] - s * vertices[i - 1][0]) / 6,
          v[1] + (s * vertices[i + 1][1] - s * vertices[i - 1][1]) / 6
        ];
        b[2] = [
          vertices[i + 1][0] +
          (s * vertices[i][0]-s * vertices[i + 2][0]) / 6,
          vertices[i + 1][1]+(s * vertices[i][1] - s*vertices[i + 2][1]) / 6
        ];
        b[3] = [
          vertices[i + 1][0],
          vertices[i + 1][1]
        ];
        this.drawingContext.bezierCurveTo(b[1][0],b[1][1],
          b[2][0],b[2][1],b[3][0],b[3][1]);
      }
      if (closeShape) {
        this.drawingContext.lineTo(vertices[i + 1][0], vertices[i + 1][1]);
      }
      this._doFillStrokeClose();
    }
  } else if (isBezier&&(shapeKind===constants.POLYGON ||shapeKind === null)) {
    this.drawingContext.beginPath();
    for (i = 0; i < numVerts; i++) {
      if (vertices[i].isVert) {
        if (vertices[i].moveTo) {
          this.drawingContext.moveTo(vertices[i][0], vertices[i][1]);
        } else {
          this.drawingContext.lineTo(vertices[i][0], vertices[i][1]);
        }
      } else {
        this.drawingContext.bezierCurveTo(vertices[i][0], vertices[i][1],
          vertices[i][2], vertices[i][3], vertices[i][4], vertices[i][5]);
      }
    }
    this._doFillStrokeClose();
  } else if (isQuadratic &&
    (shapeKind === constants.POLYGON || shapeKind === null)) {
    this.drawingContext.beginPath();
    for (i = 0; i < numVerts; i++) {
      if (vertices[i].isVert) {
        if (vertices[i].moveTo) {
          this.drawingContext.moveTo([0], vertices[i][1]);
        } else {
          this.drawingContext.lineTo(vertices[i][0], vertices[i][1]);
        }
      } else {
        this.drawingContext.quadraticCurveTo(vertices[i][0], vertices[i][1],
          vertices[i][2], vertices[i][3]);
      }
    }
    this._doFillStrokeClose();
  } else {
    if (shapeKind === constants.POINTS) {
      for (i = 0; i < numVerts; i++) {
        v = vertices[i];
        if (this._doStroke) {
          this._pInst.stroke(v[6]);
        }
        this._pInst.point(v[0], v[1]);
      }
    } else if (shapeKind === constants.LINES) {
      for (i = 0; i + 1 < numVerts; i += 2) {
        v = vertices[i];
        if (this._doStroke) {
          this._pInst.stroke(vertices[i + 1][6]);
        }
        this._pInst.line(v[0], v[1], vertices[i + 1][0], vertices[i + 1][1]);
      }
    } else if (shapeKind === constants.TRIANGLES) {
      for (i = 0; i + 2 < numVerts; i += 3) {
        v = vertices[i];
        this.drawingContext.beginPath();
        this.drawingContext.moveTo(v[0], v[1]);
        this.drawingContext.lineTo(vertices[i + 1][0], vertices[i + 1][1]);
        this.drawingContext.lineTo(vertices[i + 2][0], vertices[i + 2][1]);
        this.drawingContext.lineTo(v[0], v[1]);
        if (this._doFill) {
          this._pInst.fill(vertices[i + 2][5]);
          this.drawingContext.fill();
        }
        if (this._doStroke) {
          this._pInst.stroke(vertices[i + 2][6]);
          this.drawingContext.stroke();
        }
        this.drawingContext.closePath();
      }
    } else if (shapeKind === constants.TRIANGLE_STRIP) {
      for (i = 0; i + 1 < numVerts; i++) {
        v = vertices[i];
        this.drawingContext.beginPath();
        this.drawingContext.moveTo(vertices[i + 1][0], vertices[i + 1][1]);
        this.drawingContext.lineTo(v[0], v[1]);
        if (this._doStroke) {
          this._pInst.stroke(vertices[i + 1][6]);
        }
        if (this._doFill) {
          this._pInst.fill(vertices[i + 1][5]);
        }
        if (i + 2 < numVerts) {
          this.drawingContext.lineTo(vertices[i + 2][0], vertices[i + 2][1]);
          if (this._doStroke) {
            this._pInst.stroke(vertices[i + 2][6]);
          }
          if (this._doFill) {
            this._pInst.fill(vertices[i + 2][5]);
          }
        }
        this._doFillStrokeClose();
      }
    } else if (shapeKind === constants.TRIANGLE_FAN) {
      if (numVerts > 2) {
        this.drawingContext.beginPath();
        this.drawingContext.moveTo(vertices[0][0], vertices[0][1]);
        this.drawingContext.lineTo(vertices[1][0], vertices[1][1]);
        this.drawingContext.lineTo(vertices[2][0], vertices[2][1]);
        if (this._doFill) {
          this._pInst.fill(vertices[2][5]);
        }
        if (this._doStroke) {
          this._pInst.stroke(vertices[2][6]);
        }
        this._doFillStrokeClose();
        for (i = 3; i < numVerts; i++) {
          v = vertices[i];
          this.drawingContext.beginPath();
          this.drawingContext.moveTo(vertices[0][0], vertices[0][1]);
          this.drawingContext.lineTo(vertices[i - 1][0], vertices[i - 1][1]);
          this.drawingContext.lineTo(v[0], v[1]);
          if (this._doFill) {
            this._pInst.fill(v[5]);
          }
          if (this._doStroke) {
            this._pInst.stroke(v[6]);
          }
          this._doFillStrokeClose();
        }
      }
    } else if (shapeKind === constants.QUADS) {
      for (i = 0; i + 3 < numVerts; i += 4) {
        v = vertices[i];
        this.drawingContext.beginPath();
        this.drawingContext.moveTo(v[0], v[1]);
        for (j = 1; j < 4; j++) {
          this.drawingContext.lineTo(vertices[i + j][0], vertices[i + j][1]);
        }
        this.drawingContext.lineTo(v[0], v[1]);
        if (this._doFill) {
          this._pInst.fill(vertices[i + 3][5]);
        }
        if (this._doStroke) {
          this._pInst.stroke(vertices[i + 3][6]);
        }
        this._doFillStrokeClose();
      }
    } else if (shapeKind === constants.QUAD_STRIP) {
      if (numVerts > 3) {
        for (i = 0; i + 1 < numVerts; i += 2) {
          v = vertices[i];
          this.drawingContext.beginPath();
          if (i + 3 < numVerts) {
            this.drawingContext.moveTo(vertices[i + 2][0], vertices[i+2][1]);
            this.drawingContext.lineTo(v[0], v[1]);
            this.drawingContext.lineTo(vertices[i + 1][0], vertices[i+1][1]);
            this.drawingContext.lineTo(vertices[i + 3][0], vertices[i+3][1]);
            if (this._doFill) {
              this._pInst.fill(vertices[i + 3][5]);
            }
            if (this._doStroke) {
              this._pInst.stroke(vertices[i + 3][6]);
            }
          } else {
            this.drawingContext.moveTo(v[0], v[1]);
            this.drawingContext.lineTo(vertices[i + 1][0], vertices[i+1][1]);
          }
          this._doFillStrokeClose();
        }
      }
    } else {
      this.drawingContext.beginPath();
      this.drawingContext.moveTo(vertices[0][0], vertices[0][1]);
      for (i = 1; i < numVerts; i++) {
        v = vertices[i];
        if (v.isVert) {
          if (v.moveTo) {
            this.drawingContext.moveTo(v[0], v[1]);
          } else {
            this.drawingContext.lineTo(v[0], v[1]);
          }
        }
      }
      this._doFillStrokeClose();
    }
  }
  isCurve = false;
  isBezier = false;
  isQuadratic = false;
  isContour = false;
  if (closeShape) {
    vertices.pop();
  }
  return this;
};
//////////////////////////////////////////////
// SHAPE | Attributes
//////////////////////////////////////////////

p5.Renderer2D.prototype.noSmooth = function() {
  if ('imageSmoothingEnabled' in this.drawingContext) {
    this.drawingContext.imageSmoothingEnabled = false;
  }
  else if ('mozImageSmoothingEnabled' in this.drawingContext) {
    this.drawingContext.mozImageSmoothingEnabled = false;
  }
  else if ('webkitImageSmoothingEnabled' in this.drawingContext) {
    this.drawingContext.webkitImageSmoothingEnabled = false;
  }
  else if ('msImageSmoothingEnabled' in this.drawingContext) {
    this.drawingContext.msImageSmoothingEnabled = false;
  }
  return this;
};

p5.Renderer2D.prototype.smooth = function() {
  if ('imageSmoothingEnabled' in this.drawingContext) {
    this.drawingContext.imageSmoothingEnabled = true;
  }
  else if ('mozImageSmoothingEnabled' in this.drawingContext) {
    this.drawingContext.mozImageSmoothingEnabled = true;
  }
  else if ('webkitImageSmoothingEnabled' in this.drawingContext) {
    this.drawingContext.webkitImageSmoothingEnabled = true;
  }
  else if ('msImageSmoothingEnabled' in this.drawingContext) {
    this.drawingContext.msImageSmoothingEnabled = true;
  }
  return this;
};

p5.Renderer2D.prototype.strokeCap = function(cap) {
  if (cap === constants.ROUND ||
    cap === constants.SQUARE ||
    cap === constants.PROJECT) {
    this.drawingContext.lineCap = cap;
  }
  return this;
};

p5.Renderer2D.prototype.strokeJoin = function(join) {
  if (join === constants.ROUND ||
    join === constants.BEVEL ||
    join === constants.MITER) {
    this.drawingContext.lineJoin = join;
  }
  return this;
};

p5.Renderer2D.prototype.strokeWeight = function(w) {
  if (typeof w === 'undefined' || w === 0) {
    // hack because lineWidth 0 doesn't work
    this.drawingContext.lineWidth = 0.0001;
  } else {
    this.drawingContext.lineWidth = w;
  }
  return this;
};

p5.Renderer2D.prototype._getFill = function(){
  return this.drawingContext.fillStyle;
};

p5.Renderer2D.prototype._getStroke = function(){
  return this.drawingContext.strokeStyle;
};

//////////////////////////////////////////////
// SHAPE | Curves
//////////////////////////////////////////////
p5.Renderer2D.prototype.bezier = function (x1, y1, x2, y2, x3, y3, x4, y4) {
  this._pInst.beginShape();
  this._pInst.vertex(x1, y1);
  this._pInst.bezierVertex(x2, y2, x3, y3, x4, y4);
  this._pInst.endShape();
  return this;
};

p5.Renderer2D.prototype.curve = function (x1, y1, x2, y2, x3, y3, x4, y4) {
  this._pInst.beginShape();
  this._pInst.curveVertex(x1, y1);
  this._pInst.curveVertex(x2, y2);
  this._pInst.curveVertex(x3, y3);
  this._pInst.curveVertex(x4, y4);
  this._pInst.endShape();
  return this;
};

//////////////////////////////////////////////
// SHAPE | Vertex
//////////////////////////////////////////////

p5.Renderer2D.prototype._doFillStrokeClose = function () {
  if (this._doFill) {
    this.drawingContext.fill();
  }
  if (this._doStroke) {
    this.drawingContext.stroke();
  }
  this.drawingContext.closePath();
};

//////////////////////////////////////////////
// TRANSFORM
//////////////////////////////////////////////

p5.Renderer2D.prototype.applyMatrix =
function(n00, n01, n02, n10, n11, n12) {
  this.drawingContext.transform(n00, n01, n02, n10, n11, n12);
};

p5.Renderer2D.prototype.resetMatrix = function() {
  this.drawingContext.setTransform(1, 0, 0, 1, 0, 0);
  this.drawingContext.scale(this._pInst._pixelDensity,
                            this._pInst._pixelDensity);
  return this;
};

p5.Renderer2D.prototype.rotate = function(r) {
  this.drawingContext.rotate(r);
};

p5.Renderer2D.prototype.scale = function(x,y) {
  this.drawingContext.scale(x, y);
  return this;
};

p5.Renderer2D.prototype.shearX = function(angle) {
  if (this._pInst._angleMode === constants.DEGREES) {
    // undoing here, because it gets redone in tan()
    angle = this._pInst.degrees(angle);
  }
  this.drawingContext.transform(1, 0, this._pInst.tan(angle), 1, 0, 0);
  return this;
};

p5.Renderer2D.prototype.shearY = function(angle) {
  if (this._pInst._angleMode === constants.DEGREES) {
    // undoing here, because it gets redone in tan()
    angle = this._pInst.degrees(angle);
  }
  this.drawingContext.transform(1, this._pInst.tan(angle), 0, 1, 0, 0);
  return this;
};

p5.Renderer2D.prototype.translate = function(x, y) {
  this.drawingContext.translate(x, y);
  return this;
};

//////////////////////////////////////////////
// TYPOGRAPHY
//
//////////////////////////////////////////////

p5.Renderer2D.prototype.text = function (str, x, y, maxWidth, maxHeight) {

  var p = this._pInst, cars, n, ii, jj, line, testLine,
    testWidth, words, totalHeight, baselineHacked,
    finalMaxHeight = Number.MAX_VALUE;

  // baselineHacked: (HACK)
  // A temporary fix to conform to Processing's implementation
  // of BASELINE vertical alignment in a bounding box

  if (!(this._doFill || this._doStroke)) {
    return;
  }

  if (typeof str !== 'string') {
    str = str.toString();
  }

  str = str.replace(/(\t)/g, '  ');
  cars = str.split('\n');

  if (typeof maxWidth !== 'undefined') {

    totalHeight = 0;
    for (ii = 0; ii < cars.length; ii++) {
      line = '';
      words = cars[ii].split(' ');
      for (n = 0; n < words.length; n++) {
        testLine = line + words[n] + ' ';
        testWidth = this.textWidth(testLine);
        if (testWidth > maxWidth) {
          line = words[n] + ' ';
          totalHeight += p.textLeading();
        } else {
          line = testLine;
        }
      }
    }

    if (this._rectMode === constants.CENTER) {

      x -= maxWidth / 2;
      y -= maxHeight / 2;
    }

    switch (this.drawingContext.textAlign) {

    case constants.CENTER:
      x += maxWidth / 2;
      break;
    case constants.RIGHT:
      x += maxWidth;
      break;
    }

    if (typeof maxHeight !== 'undefined') {

      switch (this.drawingContext.textBaseline) {
      case constants.BOTTOM:
        y += (maxHeight - totalHeight);
        break;
      case constants._CTX_MIDDLE: // CENTER?
        y += (maxHeight - totalHeight) / 2;
        break;
      case constants.BASELINE:
        baselineHacked = true;
        this.drawingContext.textBaseline = constants.TOP;
        break;
      }

      // remember the max-allowed y-position for any line (fix to #928)
      finalMaxHeight = (y + maxHeight) - p.textAscent();
    }

    for (ii = 0; ii < cars.length; ii++) {

      line = '';
      words = cars[ii].split(' ');
      for (n = 0; n < words.length; n++) {
        testLine = line + words[n] + ' ';
        testWidth = this.textWidth(testLine);
        if (testWidth > maxWidth && line.length > 0) {
          this._renderText(p, line, x, y, finalMaxHeight);
          line = words[n] + ' ';
          y += p.textLeading();
        } else {
          line = testLine;
        }
      }

      this._renderText(p, line, x, y, finalMaxHeight);
      y += p.textLeading();
    }
  }
  else {
    // Offset to account for vertically centering multiple lines of text - no
    // need to adjust anything for vertical align top or baseline
    var offset = 0,
      vAlign = p.textAlign().vertical;
    if (vAlign === constants.CENTER) {
      offset = ((cars.length - 1) * p.textLeading()) / 2;
    } else if (vAlign === constants.BOTTOM) {
      offset = (cars.length - 1) * p.textLeading();
    }

    for (jj = 0; jj < cars.length; jj++) {

      this._renderText(p, cars[jj], x, y-offset, finalMaxHeight);
      y += p.textLeading();
    }
  }

  if (baselineHacked) {
    this.drawingContext.textBaseline = constants.BASELINE;
  }

  return p;
};

p5.Renderer2D.prototype._renderText = function(p, line, x, y, maxY) {

  if (y >= maxY) {
    return; // don't render lines beyond our maxY position
  }

  p.push(); // fix to #803

  if (!this._isOpenType()) {  // a system/browser font

    // no stroke unless specified by user
    if (this._doStroke && this._strokeSet) {

      this.drawingContext.strokeText(line, x, y);
    }

    if (this._doFill) {

      // if fill hasn't been set by user, use default text fill
      this.drawingContext.fillStyle =  this._fillSet ?
        this.drawingContext.fillStyle : constants._DEFAULT_TEXT_FILL;

      this.drawingContext.fillText(line, x, y);
    }
  }
  else { // an opentype font, let it handle the rendering

    this._textFont._renderPath(line, x, y, { renderer: this });
  }

  p.pop();

  return p;
};

p5.Renderer2D.prototype.textWidth = function(s) {

  if (this._isOpenType()) {

    return this._textFont._textWidth(s, this._textSize);
  }

  return this.drawingContext.measureText(s).width;
};

p5.Renderer2D.prototype.textAlign = function(h, v) {

  if (arguments.length) {

    if (h === constants.LEFT ||
      h === constants.RIGHT ||
      h === constants.CENTER) {

      this.drawingContext.textAlign = h;
    }

    if (v === constants.TOP ||
      v === constants.BOTTOM ||
      v === constants.CENTER ||
      v === constants.BASELINE) {

      if (v === constants.CENTER) {
        this.drawingContext.textBaseline = constants._CTX_MIDDLE;
      } else {
        this.drawingContext.textBaseline = v;
      }
    }

    return this._pInst;

  } else {

    var valign = this.drawingContext.textBaseline;

    if (valign === constants._CTX_MIDDLE) {

      valign = constants.CENTER;
    }

    return {

      horizontal: this.drawingContext.textAlign,
      vertical: valign
    };
  }
};

p5.Renderer2D.prototype._applyTextProperties = function() {

  var font, p = this._pInst;

  this._setProperty('_textAscent', null);
  this._setProperty('_textDescent', null);

  font = this._textFont;

  if (this._isOpenType()) {

    font = this._textFont.font.familyName;
    this._setProperty('_textStyle', this._textFont.font.styleName);
  }

  this.drawingContext.font = this._textStyle + ' ' +
  this._textSize + 'px ' + font;

  return p;
};


//////////////////////////////////////////////
// STRUCTURE
//////////////////////////////////////////////

p5.Renderer2D.prototype.push = function() {
  this.drawingContext.save();
};

p5.Renderer2D.prototype.pop = function() {
  this.drawingContext.restore();
};

module.exports = p5.Renderer2D;
},{"../image/filters":54,"./canvas":35,"./constants":36,"./core":37,"./p5.Renderer":43}],45:[function(_dereq_,module,exports){
/**
 * @module Rendering
 * @submodule Rendering
 * @for p5
 */

var p5 = _dereq_('./core');
var constants = _dereq_('./constants');
_dereq_('./p5.Graphics');
_dereq_('./p5.Renderer2D');
_dereq_('../webgl/p5.RendererGL');
var defaultId = 'defaultCanvas0'; // this gets set again in createCanvas

/**
 * Creates a canvas element in the document, and sets the dimensions of it
 * in pixels. This method should be called only once at the start of setup.
 * Calling createCanvas more than once in a sketch will result in very
 * unpredicable behavior. If you want more than one drawing canvas
 * you could use createGraphics (hidden by default but it can be shown).
 * <br><br>
 * The system variables width and height are set by the parameters passed
 * to this function. If createCanvas() is not used, the window will be
 * given a default size of 100x100 pixels.
 * <br><br>
 * For more ways to position the canvas, see the
 * <a href='https://github.com/processing/p5.js/wiki/Positioning-your-canvas'>
 * positioning the canvas</a> wiki page.
 *
 * @method createCanvas
 * @param  {Number} w width of the canvas
 * @param  {Number} h height of the canvas
 * @param  {String} [renderer] 'p2d' | 'webgl'
 * @return {Object} canvas generated
 * @example
 * <div>
 * <code>
 * function setup() {
 *   createCanvas(100, 50);
 *   background(153);
 *   line(0, 0, width, height);
 * }
 * </code>
 * </div>
 */

p5.prototype.createCanvas = function(w, h, renderer) {
  //optional: renderer, otherwise defaults to p2d
  var r = renderer || constants.P2D;
  var isDefault, c;

  //4th arg (isDefault) used when called onLoad,
  //otherwise hidden to the public api
  if(arguments[3]){
    isDefault =
    (typeof arguments[3] === 'boolean') ? arguments[3] : false;
  }

  if(r === constants.WEBGL){
    c = document.getElementById(defaultId);
    if(c){ //if defaultCanvas already exists
      c.parentNode.removeChild(c); //replace the existing defaultCanvas
    }
    c = document.createElement('canvas');
    c.id = defaultId;
  }
  else {
    if (isDefault) {
      c = document.createElement('canvas');
      var i = 0;
      while (document.getElementById('defaultCanvas'+i)) {
        i++;
      }
      defaultId = 'defaultCanvas'+i;
      c.id = defaultId;
    } else { // resize the default canvas if new one is created
      c = this.canvas;
    }
  }

  // set to invisible if still in setup (to prevent flashing with manipulate)
  if (!this._setupDone) {
    c.dataset.hidden = true; // tag to show later
    c.style.visibility='hidden';
  }

  if (this._userNode) { // user input node case
    this._userNode.appendChild(c);
  } else {
    document.body.appendChild(c);
  }



  // Init our graphics renderer
  //webgl mode
  if (r === constants.WEBGL) {
    this._setProperty('_renderer', new p5.RendererGL(c, this, true));
    this._isdefaultGraphics = true;
  }
  //P2D mode
  else {
    if (!this._isdefaultGraphics) {
      this._setProperty('_renderer', new p5.Renderer2D(c, this, true));
      this._isdefaultGraphics = true;
    }
  }
  this._renderer.resize(w, h);
  this._renderer._applyDefaults();
  if (isDefault) { // only push once
    this._elements.push(this._renderer);
  }
  return this._renderer;
};

/**
 * Resizes the canvas to given width and height. The canvas will be cleared
 * and draw will be called immediately, allowing the sketch to re-render itself
 * in the resized canvas.
 * @method resizeCanvas
 * @example
 * <div class="norender"><code>
 * function setup() {
 *   createCanvas(windowWidth, windowHeight);
 * }
 *
 * function draw() {
 *  background(0, 100, 200);
 * }
 *
 * function windowResized() {
 *   resizeCanvas(windowWidth, windowHeight);
 * }
 * </code></div>
 */
p5.prototype.resizeCanvas = function (w, h, noRedraw) {
  if (this._renderer) {

    // save canvas properties
    var props = {};
    for (var key in this.drawingContext) {
      var val = this.drawingContext[key];
      if (typeof val !== 'object' && typeof val !== 'function') {
        props[key] = val;
      }
    }
    this._renderer.resize(w, h);
    // reset canvas properties
    for (var savedKey in props) {
      this.drawingContext[savedKey] = props[savedKey];
    }
    if (!noRedraw) {
      this.redraw();
    }
  }
};


/**
 * Removes the default canvas for a p5 sketch that doesn't
 * require a canvas
 * @method noCanvas
 * @example
 * <div>
 * <code>
 * function setup() {
 *   noCanvas();
 * }
 * </code>
 * </div>
 */
p5.prototype.noCanvas = function() {
  if (this.canvas) {
    this.canvas.parentNode.removeChild(this.canvas);
  }
};

/**
 * Creates and returns a new p5.Renderer object. Use this class if you need
 * to draw into an off-screen graphics buffer. The two parameters define the
 * width and height in pixels.
 *
 * @method createGraphics
 * @param  {Number} w width of the offscreen graphics buffer
 * @param  {Number} h height of the offscreen graphics buffer
 * @param {String} renderer either 'p2d' or 'webgl'.
 * undefined defaults to p2d
 * @return {Object} offscreen graphics buffer
 * @example
 * <div>
 * <code>
 * var pg;
 * function setup() {
 *   createCanvas(100, 100);
 *   pg = createGraphics(100, 100);
 * }
 * function draw() {
 *   background(200);
 *   pg.background(100);
 *   pg.noStroke();
 *   pg.ellipse(pg.width/2, pg.height/2, 50, 50);
 *   image(pg, 50, 50);
 *   image(pg, 0, 0, 50, 50);
 * }
 * </code>
 * </div>
 */
p5.prototype.createGraphics = function(w, h, renderer){
  return new p5.Graphics(w, h, renderer, this);
};

/**
 * Blends the pixels in the display window according to the defined mode.
 * There is a choice of the following modes to blend the source pixels (A)
 * with the ones of pixels already in the display window (B):
 * <ul>
 * <li><code>BLEND</code> - linear interpolation of colours: C =
 * A*factor + B. This is the default blending mode.</li>
 * <li><code>ADD</code> - sum of A and B</li>
 * <li><code>DARKEST</code> - only the darkest colour succeeds: C =
 * min(A*factor, B).</li>
 * <li><code>LIGHTEST</code> - only the lightest colour succeeds: C =
 * max(A*factor, B).</li>
 * <li><code>DIFFERENCE</code> - subtract colors from underlying image.</li>
 * <li><code>EXCLUSION</code> - similar to <code>DIFFERENCE</code>, but less
 * extreme.</li>
 * <li><code>MULTIPLY</code> - multiply the colors, result will always be
 * darker.</li>
 * <li><code>SCREEN</code> - opposite multiply, uses inverse values of the
 * colors.</li>
 * <li><code>REPLACE</code> - the pixels entirely replace the others and
 * don't utilize alpha (transparency) values.</li>
 * <li><code>OVERLAY</code> - mix of <code>MULTIPLY</code> and <code>SCREEN
 * </code>. Multiplies dark values, and screens light values.</li>
 * <li><code>HARD_LIGHT</code> - <code>SCREEN</code> when greater than 50%
 * gray, <code>MULTIPLY</code> when lower.</li>
 * <li><code>SOFT_LIGHT</code> - mix of <code>DARKEST</code> and
 * <code>LIGHTEST</code>. Works like <code>OVERLAY</code>, but not as harsh.
 * </li>
 * <li><code>DODGE</code> - lightens light tones and increases contrast,
 * ignores darks.</li>
 * <li><code>BURN</code> - darker areas are applied, increasing contrast,
 * ignores lights.</li>
 * </ul>
 *
 * @method blendMode
 * @param  {String/Constant} mode blend mode to set for canvas
 * @example
 * <div>
 * <code>
 * blendMode(LIGHTEST);
 * strokeWeight(30);
 * stroke(80, 150, 255);
 * line(25, 25, 75, 75);
 * stroke(255, 50, 50);
 * line(75, 25, 25, 75);
 * </code>
 * </div>
 * <div>
 * <code>
 * blendMode(MULTIPLY);
 * strokeWeight(30);
 * stroke(80, 150, 255);
 * line(25, 25, 75, 75);
 * stroke(255, 50, 50);
 * line(75, 25, 25, 75);
 * </code>
 * </div>
 */
p5.prototype.blendMode = function(mode) {
  if (mode === constants.BLEND || mode === constants.DARKEST ||
    mode === constants.LIGHTEST || mode === constants.DIFFERENCE ||
    mode === constants.MULTIPLY || mode === constants.EXCLUSION ||
    mode === constants.SCREEN || mode === constants.REPLACE ||
    mode === constants.OVERLAY || mode === constants.HARD_LIGHT ||
    mode === constants.SOFT_LIGHT || mode === constants.DODGE ||
    mode === constants.BURN || mode === constants.ADD ||
    mode === constants.NORMAL) {
    this._renderer.blendMode(mode);
  } else {
    throw new Error('Mode '+mode+' not recognized.');
  }
};

module.exports = p5;

},{"../webgl/p5.RendererGL":86,"./constants":36,"./core":37,"./p5.Graphics":42,"./p5.Renderer2D":44}],46:[function(_dereq_,module,exports){

// requestAnim shim layer by Paul Irish
window.requestAnimationFrame = (function(){
  return window.requestAnimationFrame      ||
        window.webkitRequestAnimationFrame ||
        window.mozRequestAnimationFrame    ||
        window.oRequestAnimationFrame      ||
        window.msRequestAnimationFrame     ||
        function(callback, element){
          // should '60' here be framerate?
          window.setTimeout(callback, 1000 / 60);
        };
})();

// use window.performance() to get max fast and accurate time in milliseconds
window.performance = window.performance || {};
window.performance.now = (function(){
  var load_date = Date.now();
  return window.performance.now        ||
        window.performance.mozNow      ||
        window.performance.msNow       ||
        window.performance.oNow        ||
        window.performance.webkitNow   ||
        function () {
          return Date.now() - load_date;
        };
})();

/*
// http://paulirish.com/2011/requestanimationframe-for-smart-animating/
// http://my.opera.com/emoller/blog/2011/12/20/
// requestanimationframe-for-smart-er-animating
// requestAnimationFrame polyfill by Erik Möller
// fixes from Paul Irish and Tino Zijdel
(function() {
  var lastTime = 0;
  var vendors = ['ms', 'moz', 'webkit', 'o'];
  for (var x = 0; x < vendors.length && !window.requestAnimationFrame; ++x) {
    window.requestAnimationFrame =
      window[vendors[x]+'RequestAnimationFrame'];
    window.cancelAnimationFrame =
      window[vendors[x]+'CancelAnimationFrame'] ||
      window[vendors[x]+'CancelRequestAnimationFrame'];
  }

  if (!window.requestAnimationFrame) {
    window.requestAnimationFrame = function(callback, element) {
      var currTime = new Date().getTime();
      var timeToCall = Math.max(0, 16 - (currTime - lastTime));
      var id = window.setTimeout(function()
        { callback(currTime + timeToCall); }, timeToCall);
      lastTime = currTime + timeToCall;
      return id;
    };
  }

  if (!window.cancelAnimationFrame) {
    window.cancelAnimationFrame = function(id) {
      clearTimeout(id);
    };
  }
}());
*/

/**
 * shim for Uint8ClampedArray.slice
 * (allows arrayCopy to work with pixels[])
 * with thanks to http://halfpapstudios.com/blog/tag/html5-canvas/
 * Enumerable set to false to protect for...in from
 * Uint8ClampedArray.prototype pollution.
 */
(function () {
  'use strict';
  if (typeof Uint8ClampedArray !== 'undefined' &&
      !Uint8ClampedArray.prototype.slice) {
    Object.defineProperty(Uint8ClampedArray.prototype, 'slice', {
      value: Array.prototype.slice,
      writable: true, configurable: true, enumerable: false
    });
  }
}());

},{}],47:[function(_dereq_,module,exports){
/**
 * @module Structure
 * @submodule Structure
 * @for p5
 * @requires core
 */

'use strict';

var p5 = _dereq_('./core');

p5.prototype.exit = function() {
  throw 'exit() not implemented, see remove()';
};
/**
 * Stops p5.js from continuously executing the code within draw().
 * If loop() is called, the code in draw() begins to run continuously again.
 * If using noLoop() in setup(), it should be the last line inside the block.
 * <br><br>
 * When noLoop() is used, it's not possible to manipulate or access the
 * screen inside event handling functions such as mousePressed() or
 * keyPressed(). Instead, use those functions to call redraw() or loop(),
 * which will run draw(), which can update the screen properly. This means
 * that when noLoop() has been called, no drawing can happen, and functions
 * like saveFrame() or loadPixels() may not be used.
 * <br><br>
 * Note that if the sketch is resized, redraw() will be called to update
 * the sketch, even after noLoop() has been specified. Otherwise, the sketch
 * would enter an odd state until loop() was called.
 *
 * @method noLoop
 * @example
 * <div><code>
 * function setup() {
 *   createCanvas(100, 100);
 *   background(200);
 *   noLoop();
 * }

 * function draw() {
 *   line(10, 10, 90, 90);
 * }
 * </code></div>
 *
 * <div><code>
 * var x = 0;
 * function setup() {
 *   createCanvas(100, 100);
 * }
 *
 * function draw() {
 *   background(204);
 *   x = x + 0.1;
 *   if (x > width) {
 *     x = 0;
 *   }
 *   line(x, 0, x, height);
 * }
 *
 * function mousePressed() {
 *   noLoop();
 * }
 *
 * function mouseReleased() {
 *   loop();
 * }
 * </code></div>
 */
p5.prototype.noLoop = function() {
  this._loop = false;
};
/**
 * By default, p5.js loops through draw() continuously, executing the code
 * within it. However, the draw() loop may be stopped by calling noLoop().
 * In that case, the draw() loop can be resumed with loop().
 *
 * @method loop
 * @example
 * <div><code>
 * var x = 0;
 * function setup() {
 *   createCanvas(100, 100);
 *   noLoop();
 * }
 *
 * function draw() {
 *   background(204);
 *   x = x + 0.1;
 *   if (x > width) {
 *     x = 0;
 *   }
 *   line(x, 0, x, height);
 * }
 *
 * function mousePressed() {
 *   loop();
 * }
 *
 * function mouseReleased() {
 *   noLoop();
 * }
 * </code></div>
 */

p5.prototype.loop = function() {
  this._loop = true;
  this._draw();
};

/**
 * The push() function saves the current drawing style settings and
 * transformations, while pop() restores these settings. Note that these
 * functions are always used together. They allow you to change the style
 * and transformation settings and later return to what you had. When a new
 * state is started with push(), it builds on the current style and transform
 * information. The push() and pop() functions can be embedded to provide
 * more control. (See the second example for a demonstration.)
 * <br><br>
 * push() stores information related to the current transformation state
 * and style settings controlled by the following functions: fill(),
 * stroke(), tint(), strokeWeight(), strokeCap(), strokeJoin(),
 * imageMode(), rectMode(), ellipseMode(), colorMode(), textAlign(),
 * textFont(), textMode(), textSize(), textLeading().
 *
 * @method push
 * @example
 * <div>
 * <code>
 * ellipse(0, 50, 33, 33);  // Left circle
 *
 * push();  // Start a new drawing state
 * strokeWeight(10);
 * fill(204, 153, 0);
 * translate(50, 0);
 * ellipse(0, 50, 33, 33);  // Middle circle
 * pop();  // Restore original state
 *
 * ellipse(100, 50, 33, 33);  // Right circle
 * </code>
 * </div>
 * <div>
 * <code>
 * ellipse(0, 50, 33, 33);  // Left circle
 *
 * push();  // Start a new drawing state
 * strokeWeight(10);
 * fill(204, 153, 0);
 * ellipse(33, 50, 33, 33);  // Left-middle circle
 *
 * push();  // Start another new drawing state
 * stroke(0, 102, 153);
 * ellipse(66, 50, 33, 33);  // Right-middle circle
 * pop();  // Restore previous state
 *
 * pop();  // Restore original state
 *
 * ellipse(100, 50, 33, 33);  // Right circle
 * </code>
 * </div>
 */
p5.prototype.push = function () {
  this._renderer.push();
  this._styles.push({
    _doStroke: this._renderer._doStroke,
    _strokeSet: this._renderer._strokeSet,
    _doFill: this._renderer._doFill,
    _fillSet: this._renderer._fillSet,
    _tint: this._renderer._tint,
    _imageMode: this._renderer._imageMode,
    _rectMode: this._renderer._rectMode,
    _ellipseMode: this._renderer._ellipseMode,
    _colorMode: this._renderer._colorMode,
    _textFont: this._renderer._textFont,
    _textLeading: this._renderer._textLeading,
    _textSize: this._renderer._textSize,
    _textStyle: this._renderer._textStyle
  });
};

/**
 * The push() function saves the current drawing style settings and
 * transformations, while pop() restores these settings. Note that these
 * functions are always used together. They allow you to change the style
 * and transformation settings and later return to what you had. When a new
 * state is started with push(), it builds on the current style and transform
 * information. The push() and pop() functions can be embedded to provide
 * more control. (See the second example for a demonstration.)
 * <br><br>
 * push() stores information related to the current transformation state
 * and style settings controlled by the following functions: fill(),
 * stroke(), tint(), strokeWeight(), strokeCap(), strokeJoin(),
 * imageMode(), rectMode(), ellipseMode(), colorMode(), textAlign(),
 * textFont(), textMode(), textSize(), textLeading().
 *
 * @method pop
 * @example
 * <div>
 * <code>
 * ellipse(0, 50, 33, 33);  // Left circle
 *
 * push();  // Start a new drawing state
 * translate(50, 0);
 * strokeWeight(10);
 * fill(204, 153, 0);
 * ellipse(0, 50, 33, 33);  // Middle circle
 * pop();  // Restore original state
 *
 * ellipse(100, 50, 33, 33);  // Right circle
 * </code>
 * </div>
 * <div>
 * <code>
 * ellipse(0, 50, 33, 33);  // Left circle
 *
 * push();  // Start a new drawing state
 * strokeWeight(10);
 * fill(204, 153, 0);
 * ellipse(33, 50, 33, 33);  // Left-middle circle
 *
 * push();  // Start another new drawing state
 * stroke(0, 102, 153);
 * ellipse(66, 50, 33, 33);  // Right-middle circle
 * pop();  // Restore previous state
 *
 * pop();  // Restore original state
 *
 * ellipse(100, 50, 33, 33);  // Right circle
 * </code>
 * </div>
 */
p5.prototype.pop = function () {
  this._renderer.pop();
  var lastS = this._styles.pop();
  for(var prop in lastS){
    this._renderer[prop] = lastS[prop];
  }
};

p5.prototype.pushStyle = function() {
  throw new Error('pushStyle() not used, see push()');
};

p5.prototype.popStyle = function() {
  throw new Error('popStyle() not used, see pop()');
};

/**
 *
 * Executes the code within draw() one time. This functions allows the
 * program to update the display window only when necessary, for example
 * when an event registered by mousePressed() or keyPressed() occurs.
 * <br><br>
 * In structuring a program, it only makes sense to call redraw() within
 * events such as mousePressed(). This is because redraw() does not run
 * draw() immediately (it only sets a flag that indicates an update is
 * needed).
 * <br><br>
 * The redraw() function does not work properly when called inside draw().
 * To enable/disable animations, use loop() and noLoop().
 *
 * @method redraw
 * @example
 *   <div><code>
 *     var x = 0;
 *
 *     function setup() {
 *       createCanvas(100, 100);
 *       noLoop();
 *     }
 *
 *     function draw() {
 *       background(204);
 *       line(x, 0, x, height);
 *     }
 *
 *     function mousePressed() {
 *       x += 1;
 *       redraw();
 *     }
 *   </code></div>
 */
p5.prototype.redraw = function () {
  var userSetup = this.setup || window.setup;
  var userDraw = this.draw || window.draw;
  if (typeof userDraw === 'function') {
    if (typeof userSetup === 'undefined') {
      this.scale(this._pixelDensity, this._pixelDensity);
    }
    var self = this;
    this._registeredMethods.pre.forEach(function (f) {
      f.call(self);
    });
    userDraw();
    this._registeredMethods.post.forEach(function (f) {
      f.call(self);
    });
  }
};

p5.prototype.size = function() {
  var s = 'size() is not a valid p5 function, to set the size of the ';
  s += 'drawing canvas, please use createCanvas() instead';
  throw s;
};


module.exports = p5;

},{"./core":37}],48:[function(_dereq_,module,exports){
/**
 * @module Transform
 * @submodule Transform
 * @for p5
 * @requires core
 * @requires constants
 */


'use strict';

var p5 = _dereq_('./core');
var constants = _dereq_('./constants');

/**
 * Multiplies the current matrix by the one specified through the parameters.
 * This is very slow because it will try to calculate the inverse of the
 * transform, so avoid it whenever possible.
 *
 * @method applyMatrix
 * @param  {Number} n00 numbers which define the 3x2 matrix to be multiplied
 * @param  {Number} n01 numbers which define the 3x2 matrix to be multiplied
 * @param  {Number} n02 numbers which define the 3x2 matrix to be multiplied
 * @param  {Number} n10 numbers which define the 3x2 matrix to be multiplied
 * @param  {Number} n11 numbers which define the 3x2 matrix to be multiplied
 * @param  {Number} n12 numbers which define the 3x2 matrix to be multiplied
 * @return {p5}         the p5 object
 * @example
 * <div>
 * <code>
 * // Example in the works.
 * </code>
 * </div>
 */
p5.prototype.applyMatrix = function(n00, n01, n02, n10, n11, n12) {
  this._renderer.applyMatrix(n00, n01, n02, n10, n11, n12);
  return this;
};

p5.prototype.popMatrix = function() {
  throw new Error('popMatrix() not used, see pop()');
};

p5.prototype.printMatrix = function() {
  throw new Error('printMatrix() not implemented');
};

p5.prototype.pushMatrix = function() {
  throw new Error('pushMatrix() not used, see push()');
};

/**
 * Replaces the current matrix with the identity matrix.
 *
 * @method resetMatrix
 * @return {p5} the p5 object
 * @example
 * <div>
 * <code>
 * // Example in the works.
 * </code>
 * </div>
 */
p5.prototype.resetMatrix = function() {
  this._renderer.resetMatrix();
  return this;
};

/**
 * Rotates a shape the amount specified by the angle parameter. This
 * function accounts for angleMode, so angles can be entered in either
 * RADIANS or DEGREES.
 * <br><br>
 * Objects are always rotated around their relative position to the
 * origin and positive numbers rotate objects in a clockwise direction.
 * Transformations apply to everything that happens after and subsequent
 * calls to the function accumulates the effect. For example, calling
 * rotate(HALF_PI) and then rotate(HALF_PI) is the same as rotate(PI).
 * All tranformations are reset when draw() begins again.
 * <br><br>
 * Technically, rotate() multiplies the current transformation matrix
 * by a rotation matrix. This function can be further controlled by
 * the push() and pop().
 *
 * @method rotate
 * @param  {Number} angle the angle of rotation, specified in radians
 *                        or degrees, depending on current angleMode
 * @return {p5}           the p5 object
 * @example
 * <div>
 * <code>
 * translate(width/2, height/2);
 * rotate(PI/3.0);
 * rect(-26, -26, 52, 52);
 * </code>
 * </div>
 */
/**
 * @method rotate
 * @param  {Number} rad  angle in radians
 * @param  {p5.Vector | Array} axis axis to rotate around
 * @return {p5.RendererGL}      [description]
 */
p5.prototype.rotate = function() {
  var args = new Array(arguments.length);
  var r;
  for (var i = 0; i < args.length; ++i) {
    args[i] = arguments[i];
  }
  if (this._angleMode === constants.DEGREES) {
    r = this.radians(args[0]);
  } else if (this._angleMode === constants.RADIANS){
    r = args[0];
  }
  //in webgl mode
  if(args.length > 1){
    this._renderer.rotate(r, args[1]);
  }
  else {
    this._renderer.rotate(r);
  }
  return this;
};

/**
 * Rotates around X axis.
 * @method  rotateX
 * @param  {Number} rad angles in radians
 * @return {[type]}     [description]
 */
p5.prototype.rotateX = function(rad) {
  var args = new Array(arguments.length);
  for (var i = 0; i < args.length; ++i) {
    args[i] = arguments[i];
  }
  if (this._renderer.isP3D) {
    this._validateParameters(
      'rotateX',
      args,
      [
        ['Number']
      ]
    );
    this._renderer.rotateX(rad);
  } else {
    throw 'not supported in p2d. Please use webgl mode';
  }
  return this;
};

/**
 * Rotates around Y axis.
 * @method rotateY
 * @param  {Number} rad angles in radians
 * @return {[type]}     [description]
 */
p5.prototype.rotateY = function(rad) {
  if (this._renderer.isP3D) {
    var args = new Array(arguments.length);
    for (var i = 0; i < args.length; ++i) {
      args[i] = arguments[i];
    }
    this._validateParameters(
      'rotateY',
      args,
      [
        ['Number']
      ]
    );
    this._renderer.rotateY(rad);
  } else {
    throw 'not supported in p2d. Please use webgl mode';
  }
  return this;
};

/**
 * Rotates around Z axis.  Webgl mode only.
 * @method rotateZ
 * @param  {Number} rad angles in radians
 * @return {[type]}     [description]
 */
p5.prototype.rotateZ = function(rad) {
  if (this._renderer.isP3D) {
    var args = new Array(arguments.length);
    for (var i = 0; i < args.length; ++i) {
      args[i] = arguments[i];
    }
    this._validateParameters(
      'rotateZ',
      args,
      [
        ['Number']
      ]
    );
    this._renderer.rotateZ(rad);
  } else {
    throw 'not supported in p2d. Please use webgl mode';
  }
  return this;
};

/**
 * Increases or decreases the size of a shape by expanding and contracting
 * vertices. Objects always scale from their relative origin to the
 * coordinate system. Scale values are specified as decimal percentages.
 * For example, the function call scale(2.0) increases the dimension of a
 * shape by 200%.
 * <br><br>
 * Transformations apply to everything that happens after and subsequent
 * calls to the function multiply the effect. For example, calling scale(2.0)
 * and then scale(1.5) is the same as scale(3.0). If scale() is called
 * within draw(), the transformation is reset when the loop begins again.
 * <br><br>
 * Using this function with the z parameter is only available in WEBGL mode.
 * This function can be further controlled with push() and pop().
 *
 * @method scale
 * @param  {Number | p5.Vector | Array} s
 *                      percent to scale the object, or percentage to
 *                      scale the object in the x-axis if multiple arguments
 *                      are given
 * @param  {Number} [y] percent to scale the object in the y-axis
 * @param  {Number} [z] percent to scale the object in the z-axis (webgl only)
 * @return {p5}         the p5 object
 * @example
 * <div>
 * <code>
 * translate(width/2, height/2);
 * rotate(PI/3.0);
 * rect(-26, -26, 52, 52);
 * </code>
 * </div>
 *
 * <div>
 * <code>
 * rect(30, 20, 50, 50);
 * scale(0.5, 1.3);
 * rect(30, 20, 50, 50);
 * </code>
 * </div>
 */
p5.prototype.scale = function() {
  var x,y,z;
  var args = new Array(arguments.length);
  for(var i = 0; i < args.length; i++) {
    args[i] = arguments[i];
  }
  if(args[0] instanceof p5.Vector){
    x = args[0].x;
    y = args[0].y;
    z = args[0].z;
  }
  else if(args[0] instanceof Array){
    x = args[0][0];
    y = args[0][1];
    z = args[0][2] || 1;
  }
  else {
    if(args.length === 1){
      x = y = z = args[0];
    }
    else {
      x = args[0];
      y = args[1];
      z = args[2] || 1;
    }
  }

  if(this._renderer.isP3D){
    this._renderer.scale.call(this._renderer, x,y,z);
  }
  else {
    this._renderer.scale.call(this._renderer, x,y);
  }
  return this;
};

/**
 * Shears a shape around the x-axis the amount specified by the angle
 * parameter. Angles should be specified in the current angleMode.
 * Objects are always sheared around their relative position to the origin
 * and positive numbers shear objects in a clockwise direction.
 * <br><br>
 * Transformations apply to everything that happens after and subsequent
 * calls to the function accumulates the effect. For example, calling
 * shearX(PI/2) and then shearX(PI/2) is the same as shearX(PI).
 * If shearX() is called within the draw(), the transformation is reset when
 * the loop begins again.
 * <br><br>
 * Technically, shearX() multiplies the current transformation matrix by a
 * rotation matrix. This function can be further controlled by the
 * push() and pop() functions.
 *
 * @method shearX
 * @param  {Number} angle angle of shear specified in radians or degrees,
 *                        depending on current angleMode
 * @return {p5}           the p5 object
 * @example
 * <div>
 * <code>
 * translate(width/4, height/4);
 * shearX(PI/4.0);
 * rect(0, 0, 30, 30);
 * </code>
 * </div>
 */
p5.prototype.shearX = function(angle) {
  if (this._angleMode === constants.DEGREES) {
    angle = this.radians(angle);
  }
  this._renderer.shearX(angle);
  return this;
};

/**
 * Shears a shape around the y-axis the amount specified by the angle
 * parameter. Angles should be specified in the current angleMode. Objects
 * are always sheared around their relative position to the origin and
 * positive numbers shear objects in a clockwise direction.
 * <br><br>
 * Transformations apply to everything that happens after and subsequent
 * calls to the function accumulates the effect. For example, calling
 * shearY(PI/2) and then shearY(PI/2) is the same as shearY(PI). If
 * shearY() is called within the draw(), the transformation is reset when
 * the loop begins again.
 * <br><br>
 * Technically, shearY() multiplies the current transformation matrix by a
 * rotation matrix. This function can be further controlled by the
 * push() and pop() functions.
 *
 * @method shearY
 * @param  {Number} angle angle of shear specified in radians or degrees,
 *                        depending on current angleMode
 * @return {p5}           the p5 object
 * @example
 * <div>
 * <code>
 * translate(width/4, height/4);
 * shearY(PI/4.0);
 * rect(0, 0, 30, 30);
 * </code>
 * </div>
 */
p5.prototype.shearY = function(angle) {
  if (this._angleMode === constants.DEGREES) {
    angle = this.radians(angle);
  }
  this._renderer.shearY(angle);
  return this;
};

/**
 * Specifies an amount to displace objects within the display window.
 * The x parameter specifies left/right translation, the y parameter
 * specifies up/down translation.
 * <br><br>
 * Transformations are cumulative and apply to everything that happens after
 * and subsequent calls to the function accumulates the effect. For example,
 * calling translate(50, 0) and then translate(20, 0) is the same as
 * translate(70, 0). If translate() is called within draw(), the
 * transformation is reset when the loop begins again. This function can be
 * further controlled by using push() and pop().
 *
 * @method translate
 * @param  {Number} x left/right translation
 * @param  {Number} y up/down translation
 * @param  {Number} [z] forward/backward translation (webgl only)
 * @return {p5}       the p5 object
 * @example
 * <div>
 * <code>
 * translate(30, 20);
 * rect(0, 0, 55, 55);
 * </code>
 * </div>
 *
 * <div>
 * <code>
 * rect(0, 0, 55, 55);  // Draw rect at original 0,0
 * translate(30, 20);
 * rect(0, 0, 55, 55);  // Draw rect at new 0,0
 * translate(14, 14);
 * rect(0, 0, 55, 55);  // Draw rect at new 0,0
 * </code>
 * </div>
 */
p5.prototype.translate = function(x, y, z) {
  var args = new Array(arguments.length);
  for (var i = 0; i < args.length; ++i) {
    args[i] = arguments[i];
  }

  if (this._renderer.isP3D) {
    this._validateParameters(
      'translate',
      args,
      [
        //p3d
        ['Number', 'Number', 'Number']
      ]
    );
    this._renderer.translate(x, y, z);
  } else {
    this._validateParameters(
      'translate',
      args,
      [
        //p2d
        ['Number', 'Number']
      ]
    );
    this._renderer.translate(x, y);
  }
  return this;
};

module.exports = p5;

},{"./constants":36,"./core":37}],49:[function(_dereq_,module,exports){
/**
 * @module Shape
 * @submodule Vertex
 * @for p5
 * @requires core
 * @requires constants
 */

'use strict';

var p5 = _dereq_('./core');
var constants = _dereq_('./constants');
var shapeKind = null;
var vertices = [];
var contourVertices = [];
var isBezier = false;
var isCurve = false;
var isQuadratic = false;
var isContour = false;
var isFirstContour = true;

/**
 * Use the beginContour() and endContour() functions to create negative
 * shapes within shapes such as the center of the letter 'O'. beginContour()
 * begins recording vertices for the shape and endContour() stops recording.
 * The vertices that define a negative shape must "wind" in the opposite
 * direction from the exterior shape. First draw vertices for the exterior
 * clockwise order, then for internal shapes, draw vertices
 * shape in counter-clockwise.
 * <br><br>
 * These functions can only be used within a beginShape()/endShape() pair and
 * transformations such as translate(), rotate(), and scale() do not work
 * within a beginContour()/endContour() pair. It is also not possible to use
 * other shapes, such as ellipse() or rect() within.
 *
 * @method beginContour
 * @return {Object} the p5 object
 * @example
 * <div>
 * <code>
 * translate(50, 50);
 * stroke(255, 0, 0);
 * beginShape();
 * // Exterior part of shape, clockwise winding
 * vertex(-40, -40);
 * vertex(40, -40);
 * vertex(40, 40);
 * vertex(-40, 40);
 * // Interior part of shape, counter-clockwise winding
 * beginContour();
 * vertex(-20, -20);
 * vertex(-20, 20);
 * vertex(20, 20);
 * vertex(20, -20);
 * endContour();
 * endShape(CLOSE);
 * </code>
 * </div>
 */
p5.prototype.beginContour = function() {
  contourVertices = [];
  isContour = true;
  return this;
};

/**
 * Using the beginShape() and endShape() functions allow creating more
 * complex forms. beginShape() begins recording vertices for a shape and
 * endShape() stops recording. The value of the kind parameter tells it which
 * types of shapes to create from the provided vertices. With no mode
 * specified, the shape can be any irregular polygon.
 * <br><br>
 * The parameters available for beginShape() are POINTS, LINES, TRIANGLES,
 * TRIANGLE_FAN, TRIANGLE_STRIP, QUADS, and QUAD_STRIP. After calling the
 * beginShape() function, a series of vertex() commands must follow. To stop
 * drawing the shape, call endShape(). Each shape will be outlined with the
 * current stroke color and filled with the fill color.
 * <br><br>
 * Transformations such as translate(), rotate(), and scale() do not work
 * within beginShape(). It is also not possible to use other shapes, such as
 * ellipse() or rect() within beginShape().
 *
 * @method beginShape
 * @param  {Number/Constant} kind either POINTS, LINES, TRIANGLES,
 *                                TRIANGLE_FAN, TRIANGLE_STRIP, QUADS,
 *                                or QUAD_STRIP
 * @return {Object}               the p5 object
 * @example
 * <div>
 * <code>
 * beginShape();
 * vertex(30, 20);
 * vertex(85, 20);
 * vertex(85, 75);
 * vertex(30, 75);
 * endShape(CLOSE);
 * </code>
 * </div>
 *
 * <div>
 * <code>
 * // currently not working
 * beginShape(POINTS);
 * vertex(30, 20);
 * vertex(85, 20);
 * vertex(85, 75);
 * vertex(30, 75);
 * endShape();
 * </code>
 * </div>
 *
 * <div>
 * <code>
 * beginShape(LINES);
 * vertex(30, 20);
 * vertex(85, 20);
 * vertex(85, 75);
 * vertex(30, 75);
 * endShape();
 * </code>
 * </div>
 *
 * <div>
 * <code>
 * noFill();
 * beginShape();
 * vertex(30, 20);
 * vertex(85, 20);
 * vertex(85, 75);
 * vertex(30, 75);
 * endShape();
 * </code>
 * </div>
 *
 * <div>
 * <code>
 * noFill();
 * beginShape();
 * vertex(30, 20);
 * vertex(85, 20);
 * vertex(85, 75);
 * vertex(30, 75);
 * endShape(CLOSE);
 * </code>
 * </div>
 *
 * <div>
 * <code>
 * beginShape(TRIANGLES);
 * vertex(30, 75);
 * vertex(40, 20);
 * vertex(50, 75);
 * vertex(60, 20);
 * vertex(70, 75);
 * vertex(80, 20);
 * endShape();
 * </code>
 * </div>
 *
 * <div>
 * <code>
 * beginShape(TRIANGLE_STRIP);
 * vertex(30, 75);
 * vertex(40, 20);
 * vertex(50, 75);
 * vertex(60, 20);
 * vertex(70, 75);
 * vertex(80, 20);
 * vertex(90, 75);
 * endShape();
 * </code>
 * </div>
 *
 * <div>
 * <code>
 * beginShape(TRIANGLE_FAN);
 * vertex(57.5, 50);
 * vertex(57.5, 15);
 * vertex(92, 50);
 * vertex(57.5, 85);
 * vertex(22, 50);
 * vertex(57.5, 15);
 * endShape();
 * </code>
 * </div>
 *
 * <div>
 * <code>
 * beginShape(QUADS);
 * vertex(30, 20);
 * vertex(30, 75);
 * vertex(50, 75);
 * vertex(50, 20);
 * vertex(65, 20);
 * vertex(65, 75);
 * vertex(85, 75);
 * vertex(85, 20);
 * endShape();
 * </code>
 * </div>
 *
 * <div>
 * <code>
 * beginShape(QUAD_STRIP);
 * vertex(30, 20);
 * vertex(30, 75);
 * vertex(50, 20);
 * vertex(50, 75);
 * vertex(65, 20);
 * vertex(65, 75);
 * vertex(85, 20);
 * vertex(85, 75);
 * endShape();
 * </code>
 * </div>
 *
 * <div>
 * <code>
 * beginShape();
 * vertex(20, 20);
 * vertex(40, 20);
 * vertex(40, 40);
 * vertex(60, 40);
 * vertex(60, 60);
 * vertex(20, 60);
 * endShape(CLOSE);
 * </code>
 * </div>
 */
p5.prototype.beginShape = function(kind) {
  if (kind === constants.POINTS ||
    kind === constants.LINES ||
    kind === constants.TRIANGLES ||
    kind === constants.TRIANGLE_FAN ||
    kind === constants.TRIANGLE_STRIP ||
    kind === constants.QUADS ||
    kind === constants.QUAD_STRIP) {
    shapeKind = kind;
  } else {
    shapeKind = null;
  }
  if(this._renderer.isP3D){
    this._renderer.beginShape(kind);
  } else {
    vertices = [];
    contourVertices = [];
  }
  return this;
};

/**
 * Specifies vertex coordinates for Bezier curves. Each call to
 * bezierVertex() defines the position of two control points and
 * one anchor point of a Bezier curve, adding a new segment to a
 * line or shape.
 * <br><br>
 * The first time bezierVertex() is used within a
 * beginShape() call, it must be prefaced with a call to vertex()
 * to set the first anchor point. This function must be used between
 * beginShape() and endShape() and only when there is no MODE
 * parameter specified to beginShape().
 *
 * @method bezierVertex
 * @param  {Number} x2 x-coordinate for the first control point
 * @param  {Number} y2 y-coordinate for the first control point
 * @param  {Number} x3 x-coordinate for the second control point
 * @param  {Number} y3 y-coordinate for the second control point
 * @param  {Number} x4 x-coordinate for the anchor point
 * @param  {Number} y4 y-coordinate for the anchor point
 * @return {Object}    the p5 object
 * @example
 * <div>
 * <code>
 * noFill();
 * beginShape();
 * vertex(30, 20);
 * bezierVertex(80, 0, 80, 75, 30, 75);
 * endShape();
 * </code>
 * </div>
 *
 * <div>
 * <code>
 * beginShape();
 * vertex(30, 20);
 * bezierVertex(80, 0, 80, 75, 30, 75);
 * bezierVertex(50, 80, 60, 25, 30, 20);
 * endShape();
 * </code>
 * </div>
 */
p5.prototype.bezierVertex = function(x2, y2, x3, y3, x4, y4) {
  if (vertices.length === 0) {
    throw 'vertex() must be used once before calling bezierVertex()';
  } else {
    isBezier = true;
    var vert = [];
    for (var i = 0; i < arguments.length; i++) {
      vert[i] = arguments[i];
    }
    vert.isVert = false;
    if (isContour) {
      contourVertices.push(vert);
    } else {
      vertices.push(vert);
    }
  }
  return this;
};

/**
 * Specifies vertex coordinates for curves. This function may only
 * be used between beginShape() and endShape() and only when there
 * is no MODE parameter specified to beginShape().
 * <br><br>
 * The first and last points in a series of curveVertex() lines will be used to
 * guide the beginning and end of a the curve. A minimum of four
 * points is required to draw a tiny curve between the second and
 * third points. Adding a fifth point with curveVertex() will draw
 * the curve between the second, third, and fourth points. The
 * curveVertex() function is an implementation of Catmull-Rom
 * splines.
 *
 * @method curveVertex
 * @param {Number} x x-coordinate of the vertex
 * @param {Number} y y-coordinate of the vertex
 * @return {Object} the p5 object
 * @example
 * <div>
 * <code>
 * noFill();
 * beginShape();
 * curveVertex(84,  91);
 * curveVertex(84,  91);
 * curveVertex(68,  19);
 * curveVertex(21,  17);
 * curveVertex(32, 100);
 * curveVertex(32, 100);
 * endShape();
 * </code>
 * </div>
 */
p5.prototype.curveVertex = function(x,y) {
  isCurve = true;
  this.vertex(x, y);
  return this;
};

/**
 * Use the beginContour() and endContour() functions to create negative
 * shapes within shapes such as the center of the letter 'O'. beginContour()
 * begins recording vertices for the shape and endContour() stops recording.
 * The vertices that define a negative shape must "wind" in the opposite
 * direction from the exterior shape. First draw vertices for the exterior
 * clockwise order, then for internal shapes, draw vertices
 * shape in counter-clockwise.
 * <br><br>
 * These functions can only be used within a beginShape()/endShape() pair and
 * transformations such as translate(), rotate(), and scale() do not work
 * within a beginContour()/endContour() pair. It is also not possible to use
 * other shapes, such as ellipse() or rect() within.
 *
 * @method endContour
 * @return {Object} the p5 object
 * @example
 * <div>
 * <code>
 * translate(50, 50);
 * stroke(255, 0, 0);
 * beginShape();
 * // Exterior part of shape, clockwise winding
 * vertex(-40, -40);
 * vertex(40, -40);
 * vertex(40, 40);
 * vertex(-40, 40);
 * // Interior part of shape, counter-clockwise winding
 * beginContour();
 * vertex(-20, -20);
 * vertex(-20, 20);
 * vertex(20, 20);
 * vertex(20, -20);
 * endContour();
 * endShape(CLOSE);
 * </code>
 * </div>
 */
p5.prototype.endContour = function() {
  var vert = contourVertices[0].slice(); // copy all data
  vert.isVert = contourVertices[0].isVert;
  vert.moveTo = false;
  contourVertices.push(vert);

  // prevent stray lines with multiple contours
  if (isFirstContour) {
    vertices.push(vertices[0]);
    isFirstContour = false;
  }

  for (var i = 0; i < contourVertices.length; i++) {
    vertices.push(contourVertices[i]);
  }
  return this;
};

/**
 * The endShape() function is the companion to beginShape() and may only be
 * called after beginShape(). When endshape() is called, all of image data
 * defined since the previous call to beginShape() is written into the image
 * buffer. The constant CLOSE as the value for the MODE parameter to close
 * the shape (to connect the beginning and the end).
 *
 * @method endShape
 * @param  {Number/Constant} mode use CLOSE to close the shape
 * @return {Object}               the p5 object
 * @example
 * <div>
 * <code>
 * noFill();
 *
 * beginShape();
 * vertex(20, 20);
 * vertex(45, 20);
 * vertex(45, 80);
 * endShape(CLOSE);
 *
 * beginShape();
 * vertex(50, 20);
 * vertex(75, 20);
 * vertex(75, 80);
 * endShape();
 * </code>
 * </div>
 */
p5.prototype.endShape = function(mode) {
  if(this._renderer.isP3D){
    this._renderer.endShape(mode, isCurve, isBezier,
      isQuadratic, isContour, shapeKind);
  }else{
    if (vertices.length === 0) { return this; }
    if (!this._renderer._doStroke && !this._renderer._doFill) { return this; }

    var closeShape = mode === constants.CLOSE;

    // if the shape is closed, the first element is also the last element
    if (closeShape && !isContour) {
      vertices.push(vertices[0]);
    }

    this._renderer.endShape(mode, vertices, isCurve, isBezier,
      isQuadratic, isContour, shapeKind);

    // Reset some settings
    isCurve = false;
    isBezier = false;
    isQuadratic = false;
    isContour = false;
    isFirstContour = true;

    // If the shape is closed, the first element was added as last element.
    // We must remove it again to prevent the list of vertices from growing
    // over successive calls to endShape(CLOSE)
    if (closeShape) {
      vertices.pop();
    }
  }
  return this;
};

/**
 * Specifies vertex coordinates for quadratic Bezier curves. Each call to
 * quadraticVertex() defines the position of one control points and one
 * anchor point of a Bezier curve, adding a new segment to a line or shape.
 * The first time quadraticVertex() is used within a beginShape() call, it
 * must be prefaced with a call to vertex() to set the first anchor point.
 * This function must be used between beginShape() and endShape() and only
 * when there is no MODE parameter specified to beginShape().
 *
 * @method quadraticVertex
 * @param  {Number} cx x-coordinate for the control point
 * @param  {Number} cy y-coordinate for the control point
 * @param  {Number} x3 x-coordinate for the anchor point
 * @param  {Number} y3 y-coordinate for the anchor point
 * @return {Object}    the p5 object
 * @example
 * <div>
 * <code>
 * noFill();
 * strokeWeight(4);
 * beginShape();
 * vertex(20, 20);
 * quadraticVertex(80, 20, 50, 50);
 * endShape();
 * </code>
 * </div>
 *
 * <div>
 * <code>
 * noFill();
 * strokeWeight(4);
 * beginShape();
 * vertex(20, 20);
 * quadraticVertex(80, 20, 50, 50);
 * quadraticVertex(20, 80, 80, 80);
 * vertex(80, 60);
 * endShape();
 * </code>
 * </div>
 */
p5.prototype.quadraticVertex = function(cx, cy, x3, y3) {
  //if we're drawing a contour, put the points into an
  // array for inside drawing
  if(this._contourInited) {
    var pt = {};
    pt.x = cx;
    pt.y = cy;
    pt.x3 = x3;
    pt.y3 = y3;
    pt.type = constants.QUADRATIC;
    this._contourVertices.push(pt);

    return this;
  }
  if (vertices.length > 0) {
    isQuadratic = true;
    var vert = [];
    for (var i = 0; i < arguments.length; i++) {
      vert[i] = arguments[i];
    }
    vert.isVert = false;
    if (isContour) {
      contourVertices.push(vert);
    } else {
      vertices.push(vert);
    }
  } else {
    throw 'vertex() must be used once before calling quadraticVertex()';
  }
  return this;
};

/**
 * All shapes are constructed by connecting a series of vertices. vertex()
 * is used to specify the vertex coordinates for points, lines, triangles,
 * quads, and polygons. It is used exclusively within the beginShape() and
 * endShape() functions.
 *
 * @method vertex
 * @param  {Number} x x-coordinate of the vertex
 * @param  {Number} y y-coordinate of the vertex
 * @return {Object}   the p5 object
 * @example
 * <div>
 * <code>
 * beginShape(POINTS);
 * vertex(30, 20);
 * vertex(85, 20);
 * vertex(85, 75);
 * vertex(30, 75);
 * endShape();
 * </code>
 * </div>
 */
p5.prototype.vertex = function(x, y, moveTo) {
  var args = new Array(arguments.length);
  for (var i = 0; i < args.length; ++i) {
    args[i] = arguments[i];
  }
  if(this._renderer.isP3D){
    this._validateParameters(
      'vertex',
      args,
      [
        ['Number', 'Number', 'Number']
      ]
    );
    this._renderer.vertex
    (arguments[0], arguments[1], arguments[2]);
  }else{
    this._validateParameters(
      'vertex',
      args,
      [
        ['Number', 'Number'],
        ['Number', 'Number', 'Number']
      ]
    );
    var vert = [];
    vert.isVert = true;
    vert[0] = x;
    vert[1] = y;
    vert[2] = 0;
    vert[3] = 0;
    vert[4] = 0;
    vert[5] = this._renderer._getFill();
    vert[6] = this._renderer._getStroke();

    if (moveTo) {
      vert.moveTo = moveTo;
    }
    if (isContour) {
      if (contourVertices.length === 0) {
        vert.moveTo = true;
      }
      contourVertices.push(vert);
    } else {
      vertices.push(vert);
    }
  }
  return this;
};

module.exports = p5;
},{"./constants":36,"./core":37}],50:[function(_dereq_,module,exports){
/**
 * @module Events
 * @submodule Acceleration
 * @for p5
 * @requires core
 */

'use strict';

var p5 = _dereq_('../core/core');

/**
 * The system variable deviceOrientation always contains the orientation of
 * the device. The value of this variable will either be set 'landscape'
 * or 'portrait'. If no data is available it will be set to 'undefined'.
 *
 * @property deviceOrientation
 */
p5.prototype.deviceOrientation = undefined;

/**
 * The system variable accelerationX always contains the acceleration of the
 * device along the x axis. Value is represented as meters per second squared.
 *
 * @property accelerationX
 */
p5.prototype.accelerationX = 0;

/**
 * The system variable accelerationY always contains the acceleration of the
 * device along the y axis. Value is represented as meters per second squared.
 *
 * @property accelerationY
 */
p5.prototype.accelerationY = 0;

/**
 * The system variable accelerationZ always contains the acceleration of the
 * device along the z axis. Value is represented as meters per second squared.
 *
 * @property accelerationZ
 */
p5.prototype.accelerationZ = 0;

/**
 * The system variable pAccelerationX always contains the acceleration of the
 * device along the x axis in the frame previous to the current frame. Value
 * is represented as meters per second squared.
 *
 * @property pAccelerationX
 */
p5.prototype.pAccelerationX = 0;

/**
 * The system variable pAccelerationY always contains the acceleration of the
 * device along the y axis in the frame previous to the current frame. Value
 * is represented as meters per second squared.
 *
 * @property pAccelerationY
 */
p5.prototype.pAccelerationY = 0;

/**
 * The system variable pAccelerationZ always contains the acceleration of the
 * device along the z axis in the frame previous to the current frame. Value
 * is represented as meters per second squared.
 *
 * @property pAccelerationZ
 */
p5.prototype.pAccelerationZ = 0;

/**
 * _updatePAccelerations updates the pAcceleration values
 *
 * @private
 */
p5.prototype._updatePAccelerations = function(){
  this._setProperty('pAccelerationX', this.accelerationX);
  this._setProperty('pAccelerationY', this.accelerationY);
  this._setProperty('pAccelerationZ', this.accelerationZ);
};

/**
 * The system variable rotationX always contains the rotation of the
 * device along the x axis. Value is represented as 0 to +/-180 degrees.
 * <br><br>
 * Note: The order the rotations are called is important, ie. if used
 * together, it must be called in the order Z-X-Y or there might be
 * unexpected behaviour.
 *
 * @example
 * <div>
 * <code>
 * function setup(){
 *   createCanvas(100, 100, WEBGL);
 * }
 *
 * function draw(){
 *   background(200);
 *   //rotateZ(radians(rotationZ));
 *   rotateX(radians(rotationX));
 *   //rotateY(radians(rotationY));
 *   box(200, 200, 200);
 * }
 * </code>
 * </div>
 *
 * @property rotationX
 */
p5.prototype.rotationX = 0;

/**
 * The system variable rotationY always contains the rotation of the
 * device along the y axis. Value is represented as 0 to +/-90 degrees.
 * <br><br>
 * Note: The order the rotations are called is important, ie. if used
 * together, it must be called in the order Z-X-Y or there might be
 * unexpected behaviour.
 *
 * @example
 * <div>
 * <code>
 * function setup(){
 *   createCanvas(100, 100, WEBGL);
 * }
 *
 * function draw(){
 *   background(200);
 *   //rotateZ(radians(rotationZ));
 *   //rotateX(radians(rotationX));
 *   rotateY(radians(rotationY));
 *   box(200, 200, 200);
 * }
 * </code>
 * </div>
 *
 * @property rotationY
 */
p5.prototype.rotationY = 0;

/**
 * The system variable rotationZ always contains the rotation of the
 * device along the z axis. Value is represented as 0 to 359 degrees.
 * <br><br>
 * Unlike rotationX and rotationY, this variable is available for devices
 * with a built-in compass only.
 * <br><br>
 * Note: The order the rotations are called is important, ie. if used
 * together, it must be called in the order Z-X-Y or there might be
 * unexpected behaviour.
 *
 * @example
 * <div>
 * <code>
 * function setup(){
 *   createCanvas(100, 100, WEBGL);
 * }
 *
 * function draw(){
 *   background(200);
 *   rotateZ(radians(rotationZ));
 *   //rotateX(radians(rotationX));
 *   //rotateY(radians(rotationY));
 *   box(200, 200, 200);
 * }
 * </code>
 * </div>
 *
 * @property rotationZ
 */
p5.prototype.rotationZ = 0;

/**
 * The system variable pRotationX always contains the rotation of the
 * device along the x axis in the frame previous to the current frame. Value
 * is represented as 0 to +/-180 degrees.
 * <br><br>
 * pRotationX can also be used with rotationX to determine the rotate
 * direction of the device along the X-axis.
 * @example
 * <div class='norender'>
 * <code>
 * // A simple if statement looking at whether
 * // rotationX - pRotationX < 0 is true or not will be
 * // sufficient for determining the rotate direction
 * // in most cases.
 *
 * // Some extra logic is needed to account for cases where
 * // the angles wrap around.
 * var rotateDirection = 'clockwise';
 *
 * // Simple range conversion to make things simpler.
 * // This is not absolutely neccessary but the logic
 * // will be different in that case.
 *
 * var rX = rotationX + 180;
 * var pRX = pRotationX + 180;
 *
 * if ((rX - pRX > 0 && rX - pRX < 270)|| rX - pRX < -270){
 *   rotateDirection = 'clockwise';
 * } else if (rX - pRX < 0 || rX - pRX > 270){
 *   rotateDirection = 'counter-clockwise';
 * }
 * </code>
 * </div>
 *
 * @property pRotationX
 */
p5.prototype.pRotationX = 0;

/**
 * The system variable pRotationY always contains the rotation of the
 * device along the y axis in the frame previous to the current frame. Value
 * is represented as 0 to +/-90 degrees.
 * <br><br>
 * pRotationY can also be used with rotationY to determine the rotate
 * direction of the device along the Y-axis.
 * @example
 * <div class='norender'>
 * <code>
 * // A simple if statement looking at whether
 * // rotationY - pRotationY < 0 is true or not will be
 * // sufficient for determining the rotate direction
 * // in most cases.
 *
 * // Some extra logic is needed to account for cases where
 * // the angles wrap around.
 * var rotateDirection = 'clockwise';
 *
 * // Simple range conversion to make things simpler.
 * // This is not absolutely neccessary but the logic
 * // will be different in that case.
 *
 * var rY = rotationY + 180;
 * var pRY = pRotationY + 180;
 *
 * if ((rY - pRY > 0 && rY - pRY < 270)|| rY - pRY < -270){
 *   rotateDirection = 'clockwise';
 * } else if (rY - pRY < 0 || rY - pRY > 270){
 *   rotateDirection = 'counter-clockwise';
 * }
 * </code>
 * </div>
 *
 * @property pRotationY
 */
p5.prototype.pRotationY = 0;

/**
 * The system variable pRotationZ always contains the rotation of the
 * device along the z axis in the frame previous to the current frame. Value
 * is represented as 0 to 359 degrees.
 * <br><br>
 * pRotationZ can also be used with rotationZ to determine the rotate
 * direction of the device along the Z-axis.
 * @example
 * <div class='norender'>
 * <code>
 * // A simple if statement looking at whether
 * // rotationZ - pRotationZ < 0 is true or not will be
 * // sufficient for determining the rotate direction
 * // in most cases.
 *
 * // Some extra logic is needed to account for cases where
 * // the angles wrap around.
 * var rotateDirection = 'clockwise';
 *
 * if ((rotationZ - pRotationZ > 0 &&
 *   rotationZ - pRotationZ < 270)||
 *   rotationZ - pRotationZ < -270){
 *
 *   rotateDirection = 'clockwise';
 *
 * } else if (rotationZ - pRotationZ < 0 ||
 *   rotationZ - pRotationZ > 270){
 *
 *   rotateDirection = 'counter-clockwise';
 *
 * }
 * </code>
 * </div>
 *
 * @property pRotationZ
 */
p5.prototype.pRotationZ = 0;

var startAngleX = 0;
var startAngleY = 0;
var startAngleZ = 0;

var rotateDirectionX = 'clockwise';
var rotateDirectionY = 'clockwise';
var rotateDirectionZ = 'clockwise';

var pRotateDirectionX;
var pRotateDirectionY;
var pRotateDirectionZ;

p5.prototype._updatePRotations = function(){
  this._setProperty('pRotationX', this.rotationX);
  this._setProperty('pRotationY', this.rotationY);
  this._setProperty('pRotationZ', this.rotationZ);
};

p5.prototype.turnAxis = undefined;

var move_threshold = 0.5;
var shake_threshold = 30;

/**
 * The setMoveThreshold() function is used to set the movement threshold for
 * the deviceMoved() function. The default threshold is set to 0.5.
 *
 * @method setMoveThreshold
 * @param {number} value The threshold value
 */
p5.prototype.setMoveThreshold = function(val){
  if(typeof val === 'number'){
    move_threshold = val;
  }
};

/**
 * The setShakeThreshold() function is used to set the movement threshold for
 * the deviceShaken() function. The default threshold is set to 30.
 *
 * @method setShakeThreshold
 * @param {number} value The threshold value
 */
p5.prototype.setShakeThreshold = function(val){
  if(typeof val === 'number'){
    shake_threshold = val;
  }
};

/**
 * The deviceMoved() function is called when the device is moved by more than
 * the threshold value along X, Y or Z axis. The default threshold is set to
 * 0.5.
 * @method deviceMoved
 * @example
 * <div>
 * <code>
 * // Run this example on a mobile device
 * // Move the device around
 * // to change the value.
 *
 * var value = 0;
 * function draw() {
 *   fill(value);
 *   rect(25, 25, 50, 50);
 * }
 * function deviceMoved() {
 *   value = value + 5;
 *   if (value > 255) {
 *     value = 0;
 *   }
 * }
 * </code>
 * </div>
 */

/**
 * The deviceTurned() function is called when the device rotates by
 * more than 90 degrees continuously.
 * <br><br>
 * The axis that triggers the deviceTurned() method is stored in the turnAxis
 * variable. The deviceTurned() method can be locked to trigger on any axis:
 * X, Y or Z by comparing the turnAxis variable to 'X', 'Y' or 'Z'.
 *
 * @method deviceTurned
 * @example
 * <div>
 * <code>
 * // Run this example on a mobile device
 * // Rotate the device by 90 degrees
 * // to change the value.
 *
 * var value = 0;
 * function draw() {
 *   fill(value);
 *   rect(25, 25, 50, 50);
 * }
 * function deviceTurned() {
 *   if (value == 0){
 *     value = 255
 *   } else if (value == 255) {
 *     value = 0;
 *   }
 * }
 * </code>
 * </div>
 * <div>
 * <code>
 * // Run this example on a mobile device
 * // Rotate the device by 90 degrees in the
 * // X-axis to change the value.
 *
 * var value = 0;
 * function draw() {
 *   fill(value);
 *   rect(25, 25, 50, 50);
 * }
 * function deviceTurned() {
 *   if (turnAxis == 'X'){
 *     if (value == 0){
 *       value = 255
 *     } else if (value == 255) {
 *       value = 0;
 *     }
 *   }
 * }
 * </code>
 * </div>
 */

/**
 * The deviceShaken() function is called when the device total acceleration
 * changes of accelerationX and accelerationY values is more than
 * the threshold value. The default threshold is set to 30.
 * @method deviceShaken
 * @example
 * <div>
 * <code>
 * // Run this example on a mobile device
 * // Shake the device to change the value.
 *
 * var value = 0;
 * function draw() {
 *   fill(value);
 *   rect(25, 25, 50, 50);
 * }
 * function deviceShaken() {
 *   value = value + 5;
 *   if (value > 255) {
 *     value = 0;
 *   }
 * }
 * </code>
 * </div>
 */

p5.prototype._ondeviceorientation = function (e) {
  this._updatePRotations();
  this._setProperty('rotationX', e.beta);
  this._setProperty('rotationY', e.gamma);
  this._setProperty('rotationZ', e.alpha);
  this._handleMotion();
};
p5.prototype._ondevicemotion = function (e) {
  this._updatePAccelerations();
  this._setProperty('accelerationX', e.acceleration.x * 2);
  this._setProperty('accelerationY', e.acceleration.y * 2);
  this._setProperty('accelerationZ', e.acceleration.z * 2);
  this._handleMotion();
};
p5.prototype._handleMotion = function() {
  if (window.orientation === 90 || window.orientation === -90) {
    this._setProperty('deviceOrientation', 'landscape');
  } else if (window.orientation === 0) {
    this._setProperty('deviceOrientation', 'portrait');
  } else if (window.orientation === undefined) {
    this._setProperty('deviceOrientation', 'undefined');
  }
  var deviceMoved = this.deviceMoved || window.deviceMoved;
  if (typeof deviceMoved === 'function') {
    if (Math.abs(this.accelerationX - this.pAccelerationX) > move_threshold ||
      Math.abs(this.accelerationY - this.pAccelerationY) > move_threshold ||
      Math.abs(this.accelerationZ - this.pAccelerationZ) > move_threshold) {
      deviceMoved();
    }
  }
  var deviceTurned = this.deviceTurned || window.deviceTurned;
  if (typeof deviceTurned === 'function') {
    // The angles given by rotationX etc is from range -180 to 180.
    // The following will convert them to 0 to 360 for ease of calculation
    // of cases when the angles wrapped around.
    // _startAngleX will be converted back at the end and updated.
    var wRX = this.rotationX + 180;
    var wPRX = this.pRotationX + 180;
    var wSAX = startAngleX + 180;
    if ((wRX - wPRX > 0 && wRX - wPRX < 270)|| wRX - wPRX < -270){
      rotateDirectionX = 'clockwise';
    } else if (wRX - wPRX < 0 || wRX - wPRX > 270){
      rotateDirectionX = 'counter-clockwise';
    }
    if (rotateDirectionX !== pRotateDirectionX){
      wSAX = wRX;
    }
    if (Math.abs(wRX - wSAX) > 90 && Math.abs(wRX - wSAX) < 270){
      wSAX = wRX;
      this._setProperty('turnAxis', 'X');
      deviceTurned();
    }
    pRotateDirectionX = rotateDirectionX;
    startAngleX = wSAX - 180;

    // Y-axis is identical to X-axis except for changing some names.
    var wRY = this.rotationY + 180;
    var wPRY = this.pRotationY + 180;
    var wSAY = startAngleY + 180;
    if ((wRY - wPRY > 0 && wRY - wPRY < 270)|| wRY - wPRY < -270){
      rotateDirectionY = 'clockwise';
    } else if (wRY - wPRY < 0 || wRY - this.pRotationY > 270){
      rotateDirectionY = 'counter-clockwise';
    }
    if (rotateDirectionY !== pRotateDirectionY){
      wSAY = wRY;
    }
    if (Math.abs(wRY - wSAY) > 90 && Math.abs(wRY - wSAY) < 270){
      wSAY = wRY;
      this._setProperty('turnAxis', 'Y');
      deviceTurned();
    }
    pRotateDirectionY = rotateDirectionY;
    startAngleY = wSAY - 180;

    // Z-axis is already in the range 0 to 360
    // so no conversion is needed.
    if ((this.rotationZ - this.pRotationZ > 0 &&
      this.rotationZ - this.pRotationZ < 270)||
      this.rotationZ - this.pRotationZ < -270){
      rotateDirectionZ = 'clockwise';
    } else if (this.rotationZ - this.pRotationZ < 0 ||
      this.rotationZ - this.pRotationZ > 270){
      rotateDirectionZ = 'counter-clockwise';
    }
    if (rotateDirectionZ !== pRotateDirectionZ){
      startAngleZ = this.rotationZ;
    }
    if (Math.abs(this.rotationZ - startAngleZ) > 90 &&
      Math.abs(this.rotationZ - startAngleZ) < 270){
      startAngleZ = this.rotationZ;
      this._setProperty('turnAxis', 'Z');
      deviceTurned();
    }
    pRotateDirectionZ = rotateDirectionZ;
    this._setProperty('turnAxis', undefined);
  }
  var deviceShaken = this.deviceShaken || window.deviceShaken;
  if (typeof deviceShaken === 'function') {
    var accelerationChangeX;
    var accelerationChangeY;
    // Add accelerationChangeZ if acceleration change on Z is needed
    if (this.pAccelerationX !== null) {
      accelerationChangeX = Math.abs(this.accelerationX - this.pAccelerationX);
      accelerationChangeY = Math.abs(this.accelerationY - this.pAccelerationY);
    }
    if (accelerationChangeX + accelerationChangeY > shake_threshold) {
      deviceShaken();
    }
  }
};


module.exports = p5;

},{"../core/core":37}],51:[function(_dereq_,module,exports){
/**
 * @module Events
 * @submodule Keyboard
 * @for p5
 * @requires core
 */

'use strict';

var p5 = _dereq_('../core/core');

/**
 * Holds the key codes of currently pressed keys.
 * @private
 */
var downKeys = {};

/**
 * The boolean system variable keyIsPressed is true if any key is pressed
 * and false if no keys are pressed.
 *
 * @property keyIsPressed
 * @example
 * <div>
 * <code>
 * var value = 0;
 * function draw() {
 *   if (keyIsPressed === true) {
 *     fill(0);
 *   } else {
 *     fill(255);
 *   }
 *   rect(25, 25, 50, 50);
 * }
 * </code>
 * </div>
 */
p5.prototype.isKeyPressed = false;
p5.prototype.keyIsPressed = false; // khan

/**
 * The system variable key always contains the value of the most recent
 * key on the keyboard that was typed. To get the proper capitalization, it
 * is best to use it within keyTyped(). For non-ASCII keys, use the keyCode
 * variable.
 *
 * @property key
 * @example
 * <div><code>
 * // Click any key to display it!
 * // (Not Guaranteed to be Case Sensitive)
 * function setup() {
 *   fill(245, 123, 158);
 *   textSize(50);
 * }
 *
 * function draw() {
 *   background(200);
 *   text(key, 33,65); // Display last key pressed.
 * }
 * </div></code>
 */
p5.prototype.key = '';

/**
 * The variable keyCode is used to detect special keys such as BACKSPACE,
 * DELETE, ENTER, RETURN, TAB, ESCAPE, SHIFT, CONTROL, OPTION, ALT, UP_ARROW,
 * DOWN_ARROW, LEFT_ARROW, RIGHT_ARROW.
 *
 * @property keyCode
 * @example
 * <div><code>
 * var fillVal = 126;
 * function draw() {
 *   fill(fillVal);
 *   rect(25, 25, 50, 50);
 * }
 *
 * function keyPressed() {
 *   if (keyCode == UP_ARROW) {
 *     fillVal = 255;
 *   } else if (keyCode == DOWN_ARROW) {
 *     fillVal = 0;
 *   }
 *   return false; // prevent default
 * }
 * </code></div>
 */
p5.prototype.keyCode = 0;

/**
 * The keyPressed() function is called once every time a key is pressed. The
 * keyCode for the key that was pressed is stored in the keyCode variable.
 * <br><br>
 * For non-ASCII keys, use the keyCode variable. You can check if the keyCode
 * equals BACKSPACE, DELETE, ENTER, RETURN, TAB, ESCAPE, SHIFT, CONTROL,
 * OPTION, ALT, UP_ARROW, DOWN_ARROW, LEFT_ARROW, RIGHT_ARROW.
 * <br><br>
 * For ASCII keys that was pressed is stored in the key variable. However, it
 * does not distinguish between uppercase and lowercase. For this reason, it
 * is recommended to use keyTyped() to read the key variable, in which the
 * case of the variable will be distinguished.
 * <br><br>
 * Because of how operating systems handle key repeats, holding down a key
 * may cause multiple calls to keyTyped() (and keyReleased() as well). The
 * rate of repeat is set by the operating system and how each computer is
 * configured.<br><br>
 * Browsers may have different default
 * behaviors attached to various key events. To prevent any default
 * behavior for this event, add "return false" to the end of the method.
 *
 * @method keyPressed
 * @example
 * <div>
 * <code>
 * var value = 0;
 * function draw() {
 *   fill(value);
 *   rect(25, 25, 50, 50);
 * }
 * function keyPressed() {
 *   if (value === 0) {
 *     value = 255;
 *   } else {
 *     value = 0;
 *   }
 * }
 * </code>
 * </div>
 * <div>
 * <code>
 * var value = 0;
 * function draw() {
 *   fill(value);
 *   rect(25, 25, 50, 50);
 * }
 * function keyPressed() {
 *   if (keyCode === LEFT_ARROW) {
 *     value = 255;
 *   } else if (keyCode === RIGHT_ARROW) {
 *     value = 0;
 *   }
 * }
 * </code>
 * </div>
 * <div class="norender">
 * <code>
 * function keyPressed(){
 *   // Do something
 *   return false; // prevent any default behaviour
 * }
 * </code>
 * </div>
 */
p5.prototype._onkeydown = function (e) {
  if (downKeys[e.which]) { // prevent multiple firings
    return;
  }
  this._setProperty('isKeyPressed', true);
  this._setProperty('keyIsPressed', true);
  this._setProperty('keyCode', e.which);
  downKeys[e.which] = true;
  var key = String.fromCharCode(e.which);
  if (!key) {
    key = e.which;
  }
  this._setProperty('key', key);
  var keyPressed = this.keyPressed || window.keyPressed;
  if (typeof keyPressed === 'function' && !e.charCode) {
    var executeDefault = keyPressed(e);
    if(executeDefault === false) {
      e.preventDefault();
    }
  }
};
/**
 * The keyReleased() function is called once every time a key is released.
 * See key and keyCode for more information.<br><br>
 * Browsers may have different default
 * behaviors attached to various key events. To prevent any default
 * behavior for this event, add "return false" to the end of the method.
 *
 * @method keyReleased
 * @example
 * <div>
 * <code>
 * var value = 0;
 * function draw() {
 *   fill(value);
 *   rect(25, 25, 50, 50);
 * }
 * function keyReleased() {
 *   if (value === 0) {
 *     value = 255;
 *   } else {
 *     value = 0;
 *   }
 *   return false; // prevent any default behavior
 * }
 * </code>
 * </div>
 */
p5.prototype._onkeyup = function (e) {
  var keyReleased = this.keyReleased || window.keyReleased;
  this._setProperty('isKeyPressed', false);
  this._setProperty('keyIsPressed', false);
  this._setProperty('_lastKeyCodeTyped', null);
  downKeys[e.which] = false;
  //delete this._downKeys[e.which];
  var key = String.fromCharCode(e.which);
  if (!key) {
    key = e.which;
  }
  this._setProperty('key', key);
  this._setProperty('keyCode', e.which);
  if (typeof keyReleased === 'function') {
    var executeDefault = keyReleased(e);
    if(executeDefault === false) {
      e.preventDefault();
    }
  }
};

/**
 * The keyTyped() function is called once every time a key is pressed, but
 * action keys such as Ctrl, Shift, and Alt are ignored. The most recent
 * key pressed will be stored in the key variable.
 * <br><br>
 * Because of how operating systems handle key repeats, holding down a key
 * will cause multiple calls to keyTyped() (and keyReleased() as well). The
 * rate of repeat is set by the operating system and how each computer is
 * configured.<br><br>
 * Browsers may have different default behaviors attached to various key
 * events. To prevent any default behavior for this event, add "return false"
 * to the end of the method.
 *
 * @method keyTyped
 * @example
 * <div>
 * <code>
 * var value = 0;
 * function draw() {
 *   fill(value);
 *   rect(25, 25, 50, 50);
 * }
 * function keyTyped() {
 *   if (key === 'a') {
 *     value = 255;
 *   } else if (key === 'b') {
 *     value = 0;
 *   }
 *   // uncomment to prevent any default behavior
 *   // return false;
 * }
 * </code>
 * </div>
 */
p5.prototype._onkeypress = function (e) {
  if (e.which === this._lastKeyCodeTyped) { // prevent multiple firings
    return;
  }
  this._setProperty('keyCode', e.which);
  this._setProperty('_lastKeyCodeTyped', e.which); // track last keyCode
  this._setProperty('key', String.fromCharCode(e.which));
  var keyTyped = this.keyTyped || window.keyTyped;
  if (typeof keyTyped === 'function') {
    var executeDefault = keyTyped(e);
    if(executeDefault === false) {
      e.preventDefault();
    }
  }
};
/**
 * The onblur function is called when the user is no longer focused
 * on the p5 element. Because the keyup events will not fire if the user is
 * not focused on the element we must assume all keys currently down have
 * been released.
 */
p5.prototype._onblur = function (e) {
  downKeys = {};
};

/**
 * The keyIsDown() function checks if the key is currently down, i.e. pressed.
 * It can be used if you have an object that moves, and you want several keys
 * to be able to affect its behaviour simultaneously, such as moving a
 * sprite diagonally. You can put in any number representing the keyCode of
 * the key, or use any of the variable keyCode names listed
 * <a href="http://p5js.org/reference/#p5/keyCode">here</a>.
 *
 * @method keyIsDown
 * @param {Number}          code The key to check for.
 * @return {Boolean}        whether key is down or not
 * @example
 * <div><code>
 * var x = 100;
 * var y = 100;
 *
 * function setup() {
 *   createCanvas(512, 512);
 * }
 *
 * function draw() {
 *   if (keyIsDown(LEFT_ARROW))
 *     x-=5;
 *
 *   if (keyIsDown(RIGHT_ARROW))
 *     x+=5;
 *
 *   if (keyIsDown(UP_ARROW))
 *     y-=5;
 *
 *   if (keyIsDown(DOWN_ARROW))
 *     y+=5;
 *
 *   clear();
 *   fill(255, 0, 0);
 *   ellipse(x, y, 50, 50);
 * }
 * </code></div>
 */
p5.prototype.keyIsDown = function(code) {
  return downKeys[code];
};

module.exports = p5;

},{"../core/core":37}],52:[function(_dereq_,module,exports){
/**
 * @module Events
 * @submodule Mouse
 * @for p5
 * @requires core
 * @requires constants
 */


'use strict';

var p5 = _dereq_('../core/core');
var constants = _dereq_('../core/constants');

/*
 * This is a flag which is false until the first time
 * we receive a mouse event. The pmouseX and pmouseY
 * values will match the mouseX and mouseY values until
 * this interaction takes place.
 */
p5.prototype._hasMouseInteracted = false;

/**
 * The system variable mouseX always contains the current horizontal
 * position of the mouse, relative to (0, 0) of the canvas.
 *
 * @property mouseX
 *
 * @example
 * <div>
 * <code>
 * // Move the mouse across the canvas
 * function draw() {
 *   background(244, 248, 252);
 *   line(mouseX, 0, mouseX, 100);
 * }
 * </code>
 * </div>
 */
p5.prototype.mouseX = 0;

/**
 * The system variable mouseY always contains the current vertical position
 * of the mouse, relative to (0, 0) of the canvas.
 *
 * @property mouseY
 *
 * @example
 * <div>
 * <code>
 * // Move the mouse across the canvas
 * function draw() {
 *   background(244, 248, 252);
 *   line(0, mouseY, 100, mouseY);
 *}
 * </code>
 * </div>
 */
p5.prototype.mouseY = 0;

/**
 * The system variable pmouseX always contains the horizontal position of
 * the mouse in the frame previous to the current frame, relative to (0, 0)
 * of the canvas.
 *
 * @property pmouseX
 *
 * @example
 * <div>
 * <code>
 * // Move the mouse across the canvas to leave a trail
 * function setup() {
 *   //slow down the frameRate to make it more visible
 *   frameRate(10);
 * }
 *
 * function draw() {
 *   background(244, 248, 252);
 *   line(mouseX, mouseY, pmouseX, pmouseY);
 *   print(pmouseX + " -> " + mouseX);
 * }
 *
 * </code>
 * </div>
 */
p5.prototype.pmouseX = 0;

/**
 * The system variable pmouseY always contains the vertical position of the
 * mouse in the frame previous to the current frame, relative to (0, 0) of
 * the canvas.
 *
 * @property pmouseY
 *
 * @example
 * <div>
 * <code>
 * function draw() {
 *   background(237, 34, 93);
 *   fill(0);
 *   //draw a square only if the mouse is not moving
 *   if(mouseY == pmouseY && mouseX == pmouseX)
 *     rect(20,20,60,60);
 *
 *   print(pmouseY + " -> " + mouseY);
 * }
 *
 * </code>
 * </div>
 */
p5.prototype.pmouseY = 0;

/**
 * The system variable winMouseX always contains the current horizontal
 * position of the mouse, relative to (0, 0) of the window.
 *
 * @property winMouseX
 *
 * @example
 * <div>
 * <code>
 * var myCanvas;
 *
 * function setup() {
 *   //use a variable to store a pointer to the canvas
 *   myCanvas = createCanvas(100, 100);
 * }
 *
 * function draw() {
 *   background(237, 34, 93);
 *   fill(0);
 *
 *   //move the canvas to the horizontal mouse position
 *   //relative to the window
 *   myCanvas.position(winMouseX+1, windowHeight/2);
 *
 *  //the y of the square is relative to the canvas
 *  rect(20,mouseY,60,60);
 * }
 *
 * </code>
 * </div>
 */
p5.prototype.winMouseX = 0;

/**
 * The system variable winMouseY always contains the current vertical
 * position of the mouse, relative to (0, 0) of the window.
 *
 * @property winMouseY
 *
 * @example
 * <div>
 * <code>
 *var myCanvas;
 *
 * function setup() {
 *   //use a variable to store a pointer to the canvas
 *   myCanvas = createCanvas(100, 100);
 * }
 *
 * function draw() {
 *   background(237, 34, 93);
 *   fill(0);
 *
 *   //move the canvas to the vertical mouse position
 *   //relative to the window
 *   myCanvas.position(windowWidth/2, winMouseY+1);
 *
 *  //the x of the square is relative to the canvas
 *  rect(mouseX,20,60,60);
 * }
 *
 * </code>
 * </div>
 */
p5.prototype.winMouseY = 0;

/**
 * The system variable pwinMouseX always contains the horizontal position
 * of the mouse in the frame previous to the current frame, relative to
 * (0, 0) of the window.
 *
 * @property pwinMouseX
 *
 * @example
 * <div>
 * <code>
 *
 * var myCanvas;
 *
 * function setup() {
 *   //use a variable to store a pointer to the canvas
 *   myCanvas = createCanvas(100, 100);
 *   noStroke();
 *   fill(237, 34, 93);
 *   }
 *
 * function draw() {
 *   clear();
 *   //the difference between previous and
 *   //current x position is the horizontal mouse speed
 *   var speed = abs(winMouseX-pwinMouseX);
 *   //change the size of the circle
 *   //according to the horizontal speed
 *   ellipse(50, 50, 10+speed*5, 10+speed*5);
 *   //move the canvas to the mouse position
 *   myCanvas.position( winMouseX+1, winMouseY+1);
 * }
 *
 * </code>
 * </div>
 */
p5.prototype.pwinMouseX = 0;

/**
 * The system variable pwinMouseY always contains the vertical position of
 * the mouse in the frame previous to the current frame, relative to (0, 0)
 * of the window.
 *
 * @property pwinMouseY
 *
 *
 * @example
 * <div>
 * <code>
 *
 * var myCanvas;
 *
 * function setup() {
 *   //use a variable to store a pointer to the canvas
 *   myCanvas = createCanvas(100, 100);
 *   noStroke();
 *   fill(237, 34, 93);
 *   }
 *
 * function draw() {
 *   clear();
 *   //the difference between previous and
 *   //current y position is the vertical mouse speed
 *   var speed = abs(winMouseY-pwinMouseY);
 *   //change the size of the circle
 *   //according to the vertical speed
 *   ellipse(50, 50, 10+speed*5, 10+speed*5);
 *   //move the canvas to the mouse position
 *   myCanvas.position( winMouseX+1, winMouseY+1);
 * }
 *
 * </code>
 * </div>
 */
p5.prototype.pwinMouseY = 0;

/**
 * Processing automatically tracks if the mouse button is pressed and which
 * button is pressed. The value of the system variable mouseButton is either
 * LEFT, RIGHT, or CENTER depending on which button was pressed last.
 * Warning: different browsers may track mouseButton differently.
 *
 * @property mouseButton
 *
 * @example
	* <div>
	* <code>
	* function draw() {
	*   background(237, 34, 93);
	*   fill(0);
	*
	*   if (mouseIsPressed) {
	*     if (mouseButton == LEFT)
	*       ellipse(50, 50, 50, 50);
	*     if (mouseButton == RIGHT)
	*       rect(25, 25, 50, 50);
	*     if (mouseButton == CENTER)
	*       triangle(23, 75, 50, 20, 78, 75);
	*   }
	*
	*   print(mouseButton);
	* }
	* </code>
 * </div>
 */
p5.prototype.mouseButton = 0;

/**
 * The boolean system variable mouseIsPressed is true if the mouse is pressed
 * and false if not.
 *
 * @property mouseIsPressed
 *
 * @example
	* <div>
	* <code>
	* function draw() {
	*   background(237, 34, 93);
	*   fill(0);
	*
	*   if (mouseIsPressed)
	*     ellipse(50, 50, 50, 50);
	*   else
	*     rect(25, 25, 50, 50);
	*
	*   print(mouseIsPressed);
	* }
	* </code>
	* </div>
 */
p5.prototype.mouseIsPressed = false;
p5.prototype.isMousePressed = false; // both are supported

p5.prototype._updateNextMouseCoords = function(e) {
  var x = this.mouseX;
  var y = this.mouseY;
  if(e.type === 'touchstart' ||
     e.type === 'touchmove' ||
     e.type === 'touchend' || e.touches) {
    x = this.touchX;
    y = this.touchY;
  } else if(this._curElement !== null) {
    var mousePos = getMousePos(this._curElement.elt, e);
    x = mousePos.x;
    y = mousePos.y;
  }
  this._setProperty('mouseX', x);
  this._setProperty('mouseY', y);
  this._setProperty('winMouseX', e.pageX);
  this._setProperty('winMouseY', e.pageY);
  if (!this._hasMouseInteracted) {
    // For first draw, make previous and next equal
    this._updateMouseCoords();
    this._setProperty('_hasMouseInteracted', true);
  }
};

p5.prototype._updateMouseCoords = function() {
  this._setProperty('pmouseX', this.mouseX);
  this._setProperty('pmouseY', this.mouseY);
  this._setProperty('pwinMouseX', this.winMouseX);
  this._setProperty('pwinMouseY', this.winMouseY);
};

function getMousePos(canvas, evt) {
  var rect = canvas.getBoundingClientRect();
  return {
    x: evt.clientX - rect.left,
    y: evt.clientY - rect.top
  };
}

p5.prototype._setMouseButton = function(e) {
  if (e.button === 1) {
    this._setProperty('mouseButton', constants.CENTER);
  } else if (e.button === 2) {
    this._setProperty('mouseButton', constants.RIGHT);
  } else {
    this._setProperty('mouseButton', constants.LEFT);
  }
};

/**
 * The mouseMoved() function is called every time the mouse moves and a mouse
 * button is not pressed.<br><br>
 * Browsers may have different default
 * behaviors attached to various mouse events. To prevent any default
 * behavior for this event, add "return false" to the end of the method.
 *
 * @method mouseMoved
 * @example
 * <div>
 * <code>
 * // Move the mouse across the page
 * // to change its value
 *
 * var value = 0;
 * function draw() {
 *   fill(value);
 *   rect(25, 25, 50, 50);
 * }
 * function mouseMoved() {
 *   value = value + 5;
 *   if (value > 255) {
 *     value = 0;
 *   }
 * }
 * </code>
 * </div>
 *
 * <div class="norender">
 * <code>
 * function mouseMoved() {
 *   ellipse(mouseX, mouseY, 5, 5);
 *   // prevent default
 *   return false;
 * }
 * </code>
 * </div>
 */

/**
 * The mouseDragged() function is called once every time the mouse moves and
 * a mouse button is pressed. If no mouseDragged() function is defined, the
 * touchMoved() function will be called instead if it is defined.<br><br>
 * Browsers may have different default
 * behaviors attached to various mouse events. To prevent any default
 * behavior for this event, add "return false" to the end of the method.
 *
 * @method mouseDragged
 * @example
 * <div>
 * <code>
 * // Drag the mouse across the page
 * // to change its value
 *
 * var value = 0;
 * function draw() {
 *   fill(value);
 *   rect(25, 25, 50, 50);
 * }
 * function mouseDragged() {
 *   value = value + 5;
 *   if (value > 255) {
 *     value = 0;
 *   }
 * }
 * </code>
 * </div>
 *
 * <div class="norender">
 * <code>
 * function mouseDragged() {
 *   ellipse(mouseX, mouseY, 5, 5);
 *   // prevent default
 *   return false;
 * }
 * </code>
 * </div>
 */
p5.prototype._onmousemove = function(e){
  var context = this._isGlobal ? window : this;
  var executeDefault;
  this._updateNextMouseCoords(e);
  this._updateNextTouchCoords(e);
  if (!this.isMousePressed) {
    if (typeof context.mouseMoved === 'function') {
      executeDefault = context.mouseMoved(e);
      if(executeDefault === false) {
        e.preventDefault();
      }
    }
  }
  else {
    if (typeof context.mouseDragged === 'function') {
      executeDefault = context.mouseDragged(e);
      if(executeDefault === false) {
        e.preventDefault();
      }
    } else if (typeof context.touchMoved === 'function') {
      executeDefault = context.touchMoved(e);
      if(executeDefault === false) {
        e.preventDefault();
      }
    }
  }
};

/**
 * The mousePressed() function is called once after every time a mouse button
 * is pressed. The mouseButton variable (see the related reference entry)
 * can be used to determine which button has been pressed. If no
 * mousePressed() function is defined, the touchStarted() function will be
 * called instead if it is defined.<br><br>
 * Browsers may have different default
 * behaviors attached to various mouse events. To prevent any default
 * behavior for this event, add "return false" to the end of the method.
 *
 * @method mousePressed
 * @example
 * <div>
 * <code>
 * // Click within the image to change
 * // the value of the rectangle
 *
 * var value = 0;
 * function draw() {
 *   fill(value);
 *   rect(25, 25, 50, 50);
 * }
 * function mousePressed() {
 *   if (value == 0) {
 *     value = 255;
 *   } else {
 *     value = 0;
 *   }
 * }
 * </code>
 * </div>
 *
 * <div class="norender">
 * <code>
 * function mousePressed() {
 *   ellipse(mouseX, mouseY, 5, 5);
 *   // prevent default
 *   return false;
 * }
 * </code>
 * </div>
 */
p5.prototype._onmousedown = function(e) {
  var context = this._isGlobal ? window : this;
  var executeDefault;
  this._setProperty('isMousePressed', true);
  this._setProperty('mouseIsPressed', true);
  this._setMouseButton(e);
  this._updateNextMouseCoords(e);
  this._updateNextTouchCoords(e);
  if (typeof context.mousePressed === 'function') {
    executeDefault = context.mousePressed(e);
    if(executeDefault === false) {
      e.preventDefault();
    }
  } else if (typeof context.touchStarted === 'function') {
    executeDefault = context.touchStarted(e);
    if(executeDefault === false) {
      e.preventDefault();
    }
  }
};

/**
 * The mouseReleased() function is called every time a mouse button is
 * released. If no mouseReleased() function is defined, the touchEnded()
 * function will be called instead if it is defined.<br><br>
 * Browsers may have different default
 * behaviors attached to various mouse events. To prevent any default
 * behavior for this event, add "return false" to the end of the method.
 *
 *
 * @method mouseReleased
 * @example
 * <div>
 * <code>
 * // Click within the image to change
 * // the value of the rectangle
 * // after the mouse has been clicked
 *
 * var value = 0;
 * function draw() {
 *   fill(value);
 *   rect(25, 25, 50, 50);
 * }
 * function mouseReleased() {
 *   if (value == 0) {
 *     value = 255;
 *   } else {
 *     value = 0;
 *   }
 * }
 * </code>
 * </div>
 *
 * <div class="norender">
 * <code>
 * function mouseReleased() {
 *   ellipse(mouseX, mouseY, 5, 5);
 *   // prevent default
 *   return false;
 * }
 * </code>
 * </div>
 */
p5.prototype._onmouseup = function(e) {
  var context = this._isGlobal ? window : this;
  var executeDefault;
  this._setProperty('isMousePressed', false);
  this._setProperty('mouseIsPressed', false);
  if (typeof context.mouseReleased === 'function') {
    executeDefault = context.mouseReleased(e);
    if(executeDefault === false) {
      e.preventDefault();
    }
  } else if (typeof context.touchEnded === 'function') {
    executeDefault = context.touchEnded(e);
    if(executeDefault === false) {
      e.preventDefault();
    }
  }
};

p5.prototype._ondragend = p5.prototype._onmouseup;
p5.prototype._ondragover = p5.prototype._onmousemove;

/**
 * The mouseClicked() function is called once after a mouse button has been
 * pressed and then released.<br><br>
 * Browsers may have different default
 * behaviors attached to various mouse events. To prevent any default
 * behavior for this event, add "return false" to the end of the method.
 *
 * @method mouseClicked
 * @example
 * <div>
 * <code>
 * // Click within the image to change
 * // the value of the rectangle
 * // after the mouse has been clicked
 *
 * var value = 0;
 * function draw() {
 *   fill(value);
 *   rect(25, 25, 50, 50);
 * }
 * function mouseClicked() {
 *   if (value == 0) {
 *     value = 255;
 *   } else {
 *     value = 0;
 *   }
 * }
 * </code>
 * </div>
 *
 * <div class="norender">
 * <code>
 * function mouseClicked() {
 *   ellipse(mouseX, mouseY, 5, 5);
 *   // prevent default
 *   return false;
 * }
 * </code>
 * </div>
 */
p5.prototype._onclick = function(e) {
  var context = this._isGlobal ? window : this;
  if (typeof context.mouseClicked === 'function') {
    var executeDefault = context.mouseClicked(e);
    if(executeDefault === false) {
      e.preventDefault();
    }
  }
};

/**
 * The function mouseWheel() is executed every time a vertical mouse wheel
 * event is detected either triggered by an actual mouse wheel or by a
 * touchpad.<br><br>
 * The event.delta property returns the amount the mouse wheel
 * have scrolled. The values can be positive or negative depending on the
 * scroll direction (on OS X with "natural" scrolling enabled, the signs
 * are inverted).<br><br>
 * Browsers may have different default behaviors attached to various
 * mouse events. To prevent any default behavior for this event, add
 * "return false" to the end of the method.<br><br>
 * Due to the current support of the "wheel" event on Safari, the function
 * may only work as expected if "return false" is included while using Safari.
 *
 * @method mouseWheel
 *
 * @example
 * <div>
 * <code>
 * var pos = 25;
 *
 * function draw() {
 *   background(237, 34, 93);
 *   fill(0);
 *   rect(25, pos, 50, 50);
 * }
 *
 * function mouseWheel(event) {
 *   print(event.delta);
 *   //move the square according to the vertical scroll amount
 *   pos += event.delta;
 *   //uncomment to block page scrolling
 *   //return false;
 * }
 * </code>
 * </div>
 */
p5.prototype._onwheel = function(e) {
  var context = this._isGlobal ? window : this;
  if (typeof context.mouseWheel === 'function') {
    e.delta = e.deltaY;
    var executeDefault = context.mouseWheel(e);
    if(executeDefault === false) {
      e.preventDefault();
    }
  }
};

module.exports = p5;

},{"../core/constants":36,"../core/core":37}],53:[function(_dereq_,module,exports){
/**
 * @module Events
 * @submodule Touch
 * @for p5
 * @requires core
 */

'use strict';

var p5 = _dereq_('../core/core');

/*
 * This is a flag which is false until the first time
 * we receive a touch event. The ptouchX and ptouchY
 * values will match the touchX and touchY values until
 * this interaction takes place.
 */
p5.prototype._hasTouchInteracted = false;

/**
 * The system variable touchX always contains the horizontal position of
 * one finger, relative to (0, 0) of the canvas. This is best used for
 * single touch interactions. For multi-touch interactions, use the
 * touches[] array.
 *
 * @property touchX
 */
p5.prototype.touchX = 0;

/**
 * The system variable touchY always contains the vertical position of
 * one finger, relative to (0, 0) of the canvas. This is best used for
 * single touch interactions. For multi-touch interactions, use the
 * touches[] array.
 *
 * @property touchY
 */
p5.prototype.touchY = 0;

/**
 * The system variable ptouchX always contains the horizontal position of
 * one finger, relative to (0, 0) of the canvas, in the frame previous to the
 * current frame.
 *
 * @property ptouchX
 */
p5.prototype.ptouchX = 0;

/**
 * The system variable ptouchY always contains the vertical position of
 * one finger, relative to (0, 0) of the canvas, in the frame previous to the
 * current frame.
 *
 * @property ptouchY
 */
p5.prototype.ptouchY = 0;

/**
 * The system variable touches[] contains an array of the positions of all
 * current touch points, relative to (0, 0) of the canvas, and IDs identifying a
 * unique touch as it moves. Each element in the array is an object with x, y,
 * and id properties.
 *
 * @property touches[]
 */
p5.prototype.touches = [];

/**
 * The boolean system variable touchIsDown is true if the screen is
 * touched and false if not.
 *
 * @property touchIsDown
 */
p5.prototype.touchIsDown = false;

p5.prototype._updateNextTouchCoords = function(e) {
  var x = this.touchX;
  var y = this.touchY;
  if(e.type === 'mousedown' ||
     e.type === 'mousemove' ||
     e.type === 'mouseup' || !e.touches) {
    x = this.mouseX;
    y = this.mouseY;
  } else {
    if(this._curElement !== null) {
      var touchInfo = getTouchInfo(this._curElement.elt, e, 0);
      x = touchInfo.x;
      y = touchInfo.y;

      var touches = [];
      for(var i = 0; i < e.touches.length; i++){
        touches[i] = getTouchInfo(this._curElement.elt, e, i);
      }
      this._setProperty('touches', touches);
    }
  }
  this._setProperty('touchX', x);
  this._setProperty('touchY', y);
  if (!this._hasTouchInteracted) {
    // For first draw, make previous and next equal
    this._updateTouchCoords();
    this._setProperty('_hasTouchInteracted', true);
  }
};

p5.prototype._updateTouchCoords = function() {
  this._setProperty('ptouchX', this.touchX);
  this._setProperty('ptouchY', this.touchY);
};

function getTouchInfo(canvas, e, i) {
  i = i || 0;
  var rect = canvas.getBoundingClientRect();
  var touch = e.touches[i] || e.changedTouches[i];
  return {
    x: touch.clientX - rect.left,
    y: touch.clientY - rect.top,
    id: touch.identifier
  };
}

/**
 * The touchStarted() function is called once after every time a touch is
 * registered. If no touchStarted() function is defined, the mousePressed()
 * function will be called instead if it is defined.<br><br>
 * Browsers may have different default behaviors attached to various touch
 * events. To prevent any default behavior for this event, add "return false"
 * to the end of the method.
 *
 * @method touchStarted
 * @example
 * <div>
 * <code>
 * // Touch within the image to change
 * // the value of the rectangle
 *
 * var value = 0;
 * function draw() {
 *   fill(value);
 *   rect(25, 25, 50, 50);
 * }
 * function touchStarted() {
 *   if (value == 0) {
 *     value = 255;
 *   } else {
 *     value = 0;
 *   }
 * }
 * </code>
 * </div>
 *
 * <div class="norender">
 * <code>
 * function touchStarted() {
 *   ellipse(touchX, touchY, 5, 5);
 *   // prevent default
 *   return false;
 * }
 * </code>
 * </div>
 */
p5.prototype._ontouchstart = function(e) {
  var context = this._isGlobal ? window : this;
  var executeDefault;
  this._updateNextTouchCoords(e);
  this._updateNextMouseCoords(e);
  this._setProperty('touchIsDown', true);
  if(typeof context.touchStarted === 'function') {
    executeDefault = context.touchStarted(e);
    if(executeDefault === false) {
      e.preventDefault();
    }
  } else if (typeof context.mousePressed === 'function') {
    executeDefault = context.mousePressed(e);
    if(executeDefault === false) {
      e.preventDefault();
    }
    //this._setMouseButton(e);
  }
};

/**
 * The touchMoved() function is called every time a touch move is registered.
 * If no touchMoved() function is defined, the mouseDragged() function will
 * be called instead if it is defined.<br><br>
 * Browsers may have different default behaviors attached to various touch
 * events. To prevent any default behavior for this event, add "return false"
 * to the end of the method.
 *
 * @method touchMoved
 * @example
 * <div>
 * <code>
 * // Move your finger across the page
 * // to change its value
 *
 * var value = 0;
 * function draw() {
 *   fill(value);
 *   rect(25, 25, 50, 50);
 * }
 * function touchMoved() {
 *   value = value + 5;
 *   if (value > 255) {
 *     value = 0;
 *   }
 * }
 * </code>
 * </div>
 *
 * <div class="norender">
 * <code>
 * function touchMoved() {
 *   ellipse(touchX, touchY, 5, 5);
 *   // prevent default
 *   return false;
 * }
 * </code>
 * </div>
 */
p5.prototype._ontouchmove = function(e) {
  var context = this._isGlobal ? window : this;
  var executeDefault;
  this._updateNextTouchCoords(e);
  this._updateNextMouseCoords(e);
  if (typeof context.touchMoved === 'function') {
    executeDefault = context.touchMoved(e);
    if(executeDefault === false) {
      e.preventDefault();
    }
  } else if (typeof context.mouseDragged === 'function') {
    executeDefault = context.mouseDragged(e);
    if(executeDefault === false) {
      e.preventDefault();
    }
  }
};

/**
 * The touchEnded() function is called every time a touch ends. If no
 * touchEnded() function is defined, the mouseReleased() function will be
 * called instead if it is defined.<br><br>
 * Browsers may have different default behaviors attached to various touch
 * events. To prevent any default behavior for this event, add "return false"
 * to the end of the method.
 *
 * @method touchEnded
 * @example
 * <div>
 * <code>
 * // Release touch within the image to
 * // change the value of the rectangle
 *
 * var value = 0;
 * function draw() {
 *   fill(value);
 *   rect(25, 25, 50, 50);
 * }
 * function touchEnded() {
 *   if (value == 0) {
 *     value = 255;
 *   } else {
 *     value = 0;
 *   }
 * }
 * </code>
 * </div>
 *
 * <div class="norender">
 * <code>
 * function touchEnded() {
 *   ellipse(touchX, touchY, 5, 5);
 *   // prevent default
 *   return false;
 * }
 * </code>
 * </div>
 */
p5.prototype._ontouchend = function(e) {
  this._updateNextTouchCoords(e);
  this._updateNextMouseCoords(e);
  if (this.touches.length === 0) {
    this._setProperty('touchIsDown', false);
  }
  var context = this._isGlobal ? window : this;
  var executeDefault;
  if (typeof context.touchEnded === 'function') {
    executeDefault = context.touchEnded(e);
    if(executeDefault === false) {
      e.preventDefault();
    }
  } else if (typeof context.mouseReleased === 'function') {
    executeDefault = context.mouseReleased(e);
    if(executeDefault === false) {
      e.preventDefault();
    }
  }
};

module.exports = p5;

},{"../core/core":37}],54:[function(_dereq_,module,exports){
/*global ImageData:false */

/**
 * This module defines the filters for use with image buffers.
 *
 * This module is basically a collection of functions stored in an object
 * as opposed to modules. The functions are destructive, modifying
 * the passed in canvas rather than creating a copy.
 *
 * Generally speaking users of this module will use the Filters.apply method
 * on a canvas to create an effect.
 *
 * A number of functions are borrowed/adapted from
 * http://www.html5rocks.com/en/tutorials/canvas/imagefilters/
 * or the java processing implementation.
 */

'use strict';

var Filters = {};


/*
 * Helper functions
 */


/**
 * Returns the pixel buffer for a canvas
 *
 * @private
 *
 * @param  {Canvas|ImageData} canvas the canvas to get pixels from
 * @return {Uint8ClampedArray}       a one-dimensional array containing
 *                                   the data in thc RGBA order, with integer
 *                                   values between 0 and 255
 */
Filters._toPixels = function (canvas) {
  if (canvas instanceof ImageData) {
    return canvas.data;
  } else {
    return canvas.getContext('2d').getImageData(
      0,
      0,
      canvas.width,
      canvas.height
    ).data;
  }
};

/**
 * Returns a 32 bit number containing ARGB data at ith pixel in the
 * 1D array containing pixels data.
 *
 * @private
 *
 * @param  {Uint8ClampedArray} data array returned by _toPixels()
 * @param  {Integer}           i    index of a 1D Image Array
 * @return {Integer}                32 bit integer value representing
 *                                  ARGB value.
 */
Filters._getARGB = function (data, i) {
  var offset = i * 4;
  return (data[offset+3] << 24) & 0xff000000 |
    (data[offset] << 16) & 0x00ff0000 |
    (data[offset+1] << 8) & 0x0000ff00 |
    data[offset+2] & 0x000000ff;
};

/**
 * Modifies pixels RGBA values to values contained in the data object.
 *
 * @private
 *
 * @param {Uint8ClampedArray} pixels array returned by _toPixels()
 * @param {Int32Array}        data   source 1D array where each value
 *                                   represents ARGB values
 */
Filters._setPixels = function (pixels, data) {
  var offset = 0;
  for( var i = 0, al = pixels.length; i < al; i++) {
    offset = i*4;
    pixels[offset + 0] = (data[i] & 0x00ff0000)>>>16;
    pixels[offset + 1] = (data[i] & 0x0000ff00)>>>8;
    pixels[offset + 2] = (data[i] & 0x000000ff);
    pixels[offset + 3] = (data[i] & 0xff000000)>>>24;
  }
};

/**
 * Returns the ImageData object for a canvas
 * https://developer.mozilla.org/en-US/docs/Web/API/ImageData
 *
 * @private
 *
 * @param  {Canvas|ImageData} canvas canvas to get image data from
 * @return {ImageData}               Holder of pixel data (and width and
 *                                   height) for a canvas
 */
Filters._toImageData = function (canvas) {
  if (canvas instanceof ImageData) {
    return canvas;
  } else {
    return canvas.getContext('2d').getImageData(
      0,
      0,
      canvas.width,
      canvas.height
    );
  }
};

/**
 * Returns a blank ImageData object.
 *
 * @private
 *
 * @param  {Integer} width
 * @param  {Integer} height
 * @return {ImageData}
 */
Filters._createImageData = function (width, height) {
  Filters._tmpCanvas = document.createElement('canvas');
  Filters._tmpCtx = Filters._tmpCanvas.getContext('2d');
  return this._tmpCtx.createImageData(width, height);
};


/**
 * Applys a filter function to a canvas.
 *
 * The difference between this and the actual filter functions defined below
 * is that the filter functions generally modify the pixel buffer but do
 * not actually put that data back to the canvas (where it would actually
 * update what is visible). By contrast this method does make the changes
 * actually visible in the canvas.
 *
 * The apply method is the method that callers of this module would generally
 * use. It has been separated from the actual filters to support an advanced
 * use case of creating a filter chain that executes without actually updating
 * the canvas in between everystep.
 *
 * @param  {[type]} func   [description]
 * @param  {[type]} canvas [description]
 * @param  {[type]} level  [description]
 * @return {[type]}        [description]
 */
Filters.apply = function (canvas, func, filterParam) {
  var ctx = canvas.getContext('2d');
  var imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);

  //Filters can either return a new ImageData object, or just modify
  //the one they received.
  var newImageData = func(imageData, filterParam);
  if (newImageData instanceof ImageData) {
    ctx.putImageData(newImageData, 0, 0, 0, 0, canvas.width, canvas.height);
  } else {
    ctx.putImageData(imageData, 0, 0, 0, 0, canvas.width, canvas.height);
  }
};


/*
 * Filters
 */


/**
 * Converts the image to black and white pixels depending if they are above or
 * below the threshold defined by the level parameter. The parameter must be
 * between 0.0 (black) and 1.0 (white). If no level is specified, 0.5 is used.
 *
 * Borrowed from http://www.html5rocks.com/en/tutorials/canvas/imagefilters/
 *
 * @param  {Canvas} canvas
 * @param  {Float} level
 */
Filters.threshold = function (canvas, level) {
  var pixels = Filters._toPixels(canvas);

  if (level === undefined) {
    level = 0.5;
  }
  var thresh = Math.floor(level * 255);

  for (var i = 0; i < pixels.length; i += 4) {
    var r = pixels[i];
    var g = pixels[i + 1];
    var b = pixels[i + 2];
    var gray = (0.2126 * r + 0.7152 * g + 0.0722 * b);
    var val;
    if (gray >= thresh) {
      val = 255;
    } else {
      val = 0;
    }
    pixels[i] = pixels[i + 1] = pixels[i + 2] = val;
  }

};


/**
 * Converts any colors in the image to grayscale equivalents.
 * No parameter is used.
 *
 * Borrowed from http://www.html5rocks.com/en/tutorials/canvas/imagefilters/
 *
 * @param {Canvas} canvas
 */
Filters.gray = function (canvas) {
  var pixels = Filters._toPixels(canvas);

  for (var i = 0; i < pixels.length; i += 4) {
    var r = pixels[i];
    var g = pixels[i + 1];
    var b = pixels[i + 2];

    // CIE luminance for RGB
    var gray = (0.2126 * r + 0.7152 * g + 0.0722 * b);
    pixels[i] = pixels[i + 1] = pixels[i + 2] = gray;
  }
};

/**
 * Sets the alpha channel to entirely opaque. No parameter is used.
 *
 * @param {Canvas} canvas
 */
Filters.opaque = function (canvas) {
  var pixels = Filters._toPixels(canvas);

  for (var i = 0; i < pixels.length; i += 4) {
    pixels[i + 3] = 255;
  }

  return pixels;
};

/**
 * Sets each pixel to its inverse value. No parameter is used.
 * @param {Invert}
 */
Filters.invert = function (canvas) {
  var pixels = Filters._toPixels(canvas);

  for (var i = 0; i < pixels.length; i += 4) {
    pixels[i] = 255 - pixels[i];
    pixels[i + 1] = 255 - pixels[i + 1];
    pixels[i + 2] = 255 - pixels[i + 2];
  }

};


/**
 * Limits each channel of the image to the number of colors specified as
 * the parameter. The parameter can be set to values between 2 and 255, but
 * results are most noticeable in the lower ranges.
 *
 * Adapted from java based processing implementation
 *
 * @param  {Canvas} canvas
 * @param  {Integer} level
 */
Filters.posterize = function (canvas, level) {
  var pixels = Filters._toPixels(canvas);

  if ((level < 2) || (level > 255)) {
    throw new Error(
      'Level must be greater than 2 and less than 255 for posterize'
    );
  }

  var levels1 = level - 1;
  for (var i = 0; i < pixels.length; i+=4) {
    var rlevel = pixels[i];
    var glevel = pixels[i + 1];
    var blevel = pixels[i + 2];

    pixels[i] = (((rlevel * level) >> 8) * 255) / levels1;
    pixels[i + 1] = (((glevel * level) >> 8) * 255) / levels1;
    pixels[i + 2] = (((blevel * level) >> 8) * 255) / levels1;
  }
};

/**
 * reduces the bright areas in an image
 * @param  {Canvas} canvas
 *
 */
Filters.dilate = function (canvas) {
  var pixels = Filters._toPixels(canvas);
  var currIdx = 0;
  var maxIdx = pixels.length ? pixels.length/4 : 0;
  var out = new Int32Array(maxIdx);
  var currRowIdx, maxRowIdx, colOrig, colOut, currLum;
  var idxRight, idxLeft, idxUp, idxDown,
      colRight, colLeft, colUp, colDown,
      lumRight, lumLeft, lumUp, lumDown;

  while(currIdx < maxIdx) {
    currRowIdx = currIdx;
    maxRowIdx = currIdx + canvas.width;
    while (currIdx < maxRowIdx) {
      colOrig = colOut = Filters._getARGB(pixels, currIdx);
      idxLeft = currIdx - 1;
      idxRight = currIdx + 1;
      idxUp = currIdx - canvas.width;
      idxDown = currIdx + canvas.width;

      if (idxLeft < currRowIdx) {
        idxLeft = currIdx;
      }
      if (idxRight >= maxRowIdx) {
        idxRight = currIdx;
      }
      if (idxUp < 0){
        idxUp = 0;
      }
      if (idxDown >= maxIdx) {
        idxDown = currIdx;
      }
      colUp = Filters._getARGB(pixels, idxUp);
      colLeft = Filters._getARGB(pixels, idxLeft);
      colDown = Filters._getARGB(pixels, idxDown);
      colRight = Filters._getARGB(pixels, idxRight);

      //compute luminance
      currLum = 77*(colOrig>>16&0xff) +
        151*(colOrig>>8&0xff) +
        28*(colOrig&0xff);
      lumLeft = 77*(colLeft>>16&0xff) +
        151*(colLeft>>8&0xff) +
        28*(colLeft&0xff);
      lumRight = 77*(colRight>>16&0xff) +
        151*(colRight>>8&0xff) +
        28*(colRight&0xff);
      lumUp = 77*(colUp>>16&0xff) +
        151*(colUp>>8&0xff) +
        28*(colUp&0xff);
      lumDown = 77*(colDown>>16&0xff) +
        151*(colDown>>8&0xff) +
        28*(colDown&0xff);

      if (lumLeft > currLum) {
        colOut = colLeft;
        currLum = lumLeft;
      }
      if (lumRight > currLum) {
        colOut = colRight;
        currLum = lumRight;
      }
      if (lumUp > currLum) {
        colOut = colUp;
        currLum = lumUp;
      }
      if (lumDown > currLum) {
        colOut = colDown;
        currLum = lumDown;
      }
      out[currIdx++]=colOut;
    }
  }
  Filters._setPixels(pixels, out);
};

/**
 * increases the bright areas in an image
 * @param  {Canvas} canvas
 *
 */
Filters.erode = function(canvas) {
  var pixels = Filters._toPixels(canvas);
  var currIdx = 0;
  var maxIdx = pixels.length ? pixels.length/4 : 0;
  var out = new Int32Array(maxIdx);
  var currRowIdx, maxRowIdx, colOrig, colOut, currLum;
  var idxRight, idxLeft, idxUp, idxDown,
      colRight, colLeft, colUp, colDown,
      lumRight, lumLeft, lumUp, lumDown;

  while(currIdx < maxIdx) {
    currRowIdx = currIdx;
    maxRowIdx = currIdx + canvas.width;
    while (currIdx < maxRowIdx) {
      colOrig = colOut = Filters._getARGB(pixels, currIdx);
      idxLeft = currIdx - 1;
      idxRight = currIdx + 1;
      idxUp = currIdx - canvas.width;
      idxDown = currIdx + canvas.width;

      if (idxLeft < currRowIdx) {
        idxLeft = currIdx;
      }
      if (idxRight >= maxRowIdx) {
        idxRight = currIdx;
      }
      if (idxUp < 0) {
        idxUp = 0;
      }
      if (idxDown >= maxIdx) {
        idxDown = currIdx;
      }
      colUp = Filters._getARGB(pixels, idxUp);
      colLeft = Filters._getARGB(pixels, idxLeft);
      colDown = Filters._getARGB(pixels, idxDown);
      colRight = Filters._getARGB(pixels, idxRight);

      //compute luminance
      currLum = 77*(colOrig>>16&0xff) +
        151*(colOrig>>8&0xff) +
        28*(colOrig&0xff);
      lumLeft = 77*(colLeft>>16&0xff) +
        151*(colLeft>>8&0xff) +
        28*(colLeft&0xff);
      lumRight = 77*(colRight>>16&0xff) +
        151*(colRight>>8&0xff) +
        28*(colRight&0xff);
      lumUp = 77*(colUp>>16&0xff) +
        151*(colUp>>8&0xff) +
        28*(colUp&0xff);
      lumDown = 77*(colDown>>16&0xff) +
        151*(colDown>>8&0xff) +
        28*(colDown&0xff);

      if (lumLeft < currLum) {
        colOut = colLeft;
        currLum = lumLeft;
      }
      if (lumRight < currLum) {
        colOut = colRight;
        currLum = lumRight;
      }
      if (lumUp < currLum) {
        colOut = colUp;
        currLum = lumUp;
      }
      if (lumDown < currLum) {
        colOut = colDown;
        currLum = lumDown;
      }

      out[currIdx++]=colOut;
    }
  }
  Filters._setPixels(pixels, out);
};

// BLUR

// internal kernel stuff for the gaussian blur filter
var blurRadius;
var blurKernelSize;
var blurKernel;
var blurMult;

/*
 * Port of https://github.com/processing/processing/blob/
 * master/core/src/processing/core/PImage.java#L1250
 *
 * Optimized code for building the blur kernel.
 * further optimized blur code (approx. 15% for radius=20)
 * bigger speed gains for larger radii (~30%)
 * added support for various image types (ALPHA, RGB, ARGB)
 * [toxi 050728]
 */
function buildBlurKernel(r) {
  var radius = (r * 3.5)|0;
  radius = (radius < 1) ? 1 : ((radius < 248) ? radius : 248);

  if (blurRadius !== radius) {
    blurRadius = radius;
    blurKernelSize = 1 + blurRadius<<1;
    blurKernel = new Int32Array(blurKernelSize);
    blurMult = new Array(blurKernelSize);
    for(var l = 0; l < blurKernelSize; l++){
      blurMult[l] = new Int32Array(256);
    }

    var bk,bki;
    var bm,bmi;

    for (var i = 1, radiusi = radius - 1; i < radius; i++) {
      blurKernel[radius+i] = blurKernel[radiusi] = bki = radiusi * radiusi;
      bm = blurMult[radius+i];
      bmi = blurMult[radiusi--];
      for (var j = 0; j < 256; j++){
        bm[j] = bmi[j] = bki * j;
      }
    }
    bk = blurKernel[radius] = radius * radius;
    bm = blurMult[radius];

    for (var k = 0; k < 256; k++){
      bm[k] = bk * k;
    }
  }

}

// Port of https://github.com/processing/processing/blob/
// master/core/src/processing/core/PImage.java#L1433
function blurARGB(canvas, radius) {
  var pixels = Filters._toPixels(canvas);
  var width = canvas.width;
  var height = canvas.height;
  var numPackedPixels = width * height;
  var argb = new Int32Array(numPackedPixels);
  for (var j = 0; j < numPackedPixels; j++) {
    argb[j] = Filters._getARGB(pixels, j);
  }
  var sum, cr, cg, cb, ca;
  var read, ri, ym, ymi, bk0;
  var a2 = new Int32Array(numPackedPixels);
  var r2 = new Int32Array(numPackedPixels);
  var g2 = new Int32Array(numPackedPixels);
  var b2 = new Int32Array(numPackedPixels);
  var yi = 0;
  buildBlurKernel(radius);
  var x, y, i;
  var bm;
  for (y = 0; y < height; y++) {
    for (x = 0; x < width; x++) {
      cb = cg = cr = ca = sum = 0;
      read = x - blurRadius;
      if (read < 0) {
        bk0 = -read;
        read = 0;
      } else {
        if (read >= width) {
          break;
        }
        bk0 = 0;
      }
      for (i = bk0; i < blurKernelSize; i++) {
        if (read >= width) {
          break;
        }
        var c = argb[read + yi];
        bm = blurMult[i];
        ca += bm[(c & -16777216) >>> 24];
        cr += bm[(c & 16711680) >> 16];
        cg += bm[(c & 65280) >> 8];
        cb += bm[c & 255];
        sum += blurKernel[i];
        read++;
      }
      ri = yi + x;
      a2[ri] = ca / sum;
      r2[ri] = cr / sum;
      g2[ri] = cg / sum;
      b2[ri] = cb / sum;
    }
    yi += width;
  }
  yi = 0;
  ym = -blurRadius;
  ymi = ym * width;
  for (y = 0; y < height; y++) {
    for (x = 0; x < width; x++) {
      cb = cg = cr = ca = sum = 0;
      if (ym < 0) {
        bk0 = ri = -ym;
        read = x;
      } else {
        if (ym >= height) {
          break;
        }
        bk0 = 0;
        ri = ym;
        read = x + ymi;
      }
      for (i = bk0; i < blurKernelSize; i++) {
        if (ri >= height) {
          break;
        }
        bm = blurMult[i];
        ca += bm[a2[read]];
        cr += bm[r2[read]];
        cg += bm[g2[read]];
        cb += bm[b2[read]];
        sum += blurKernel[i];
        ri++;
        read += width;
      }
      argb[x + yi] = (ca/sum)<<24 | (cr/sum)<<16 | (cg/sum)<<8 | (cb/sum);
    }
    yi += width;
    ymi += width;
    ym++;
  }
  Filters._setPixels(pixels, argb);
}

Filters.blur = function(canvas, radius){
  blurARGB(canvas, radius);
};


module.exports = Filters;

},{}],55:[function(_dereq_,module,exports){
/**
 * @module Image
 * @submodule Image
 * @for p5
 * @requires core
 */

/**
 * This module defines the p5 methods for the p5.Image class
 * for drawing images to the main display canvas.
 */
'use strict';


var p5 = _dereq_('../core/core');

/* global frames:true */// This is not global, but JSHint is not aware that
// this module is implicitly enclosed with Browserify: this overrides the
// redefined-global error and permits using the name "frames" for the array
// of saved animation frames.
var frames = [];


/**
 * Creates a new p5.Image (the datatype for storing images). This provides a
 * fresh buffer of pixels to play with. Set the size of the buffer with the
 * width and height parameters.
 * <br><br>
 * .pixels gives access to an array containing the values for all the pixels
 * in the display window.
 * These values are numbers. This array is the size (including an appropriate
 * factor for the pixelDensity) of the display window x4,
 * representing the R, G, B, A values in order for each pixel, moving from
 * left to right across each row, then down each column. See .pixels for
 * more info. It may also be simpler to use set() or get().
 * <br><br>
 * Before accessing the pixels of an image, the data must loaded with the
 * loadPixels() function. After the array data has been modified, the
 * updatePixels() function must be run to update the changes.
 *
 * @method createImage
 * @param  {Integer} width  width in pixels
 * @param  {Integer} height height in pixels
 * @return {p5.Image}       the p5.Image object
 * @example
 * <div>
 * <code>
 * img = createImage(66, 66);
 * img.loadPixels();
 * for (i = 0; i < img.width; i++) {
 *   for (j = 0; j < img.height; j++) {
 *     img.set(i, j, color(0, 90, 102));
 *   }
 * }
 * img.updatePixels();
 * image(img, 17, 17);
 * </code>
 * </div>
 *
 * <div>
 * <code>
 * img = createImage(66, 66);
 * img.loadPixels();
 * for (i = 0; i < img.width; i++) {
 *   for (j = 0; j < img.height; j++) {
 *     img.set(i, j, color(0, 90, 102, i % img.width * 2));
 *   }
 * }
 * img.updatePixels();
 * image(img, 17, 17);
 * image(img, 34, 34);
 * </code>
 * </div>
 *
 * <div>
 * <code>
 * var pink = color(255, 102, 204);
 * img = createImage(66, 66);
 * img.loadPixels();
 * var d = pixelDensity;
 * var halfImage = 4 * (width * d) * (height/2 * d);
 * for (var i = 0; i < halfImage; i+=4) {
 *   img.pixels[i] = red(pink);
 *   img.pixels[i+1] = green(pink);
 *   img.pixels[i+2] = blue(pink);
 *   img.pixels[i+3] = alpha(pink);
 * }
 * img.updatePixels();
 * image(img, 17, 17);
 * </code>
 * </div>
 */
p5.prototype.createImage = function(width, height) {
  return new p5.Image(width, height);
};

/**
 *  Save the current canvas as an image. In Safari, this will open the
 *  image in the window and the user must provide their own
 *  filename on save-as. Other browsers will either save the
 *  file immediately, or prompt the user with a dialogue window.
 *
 *  @method saveCanvas
 *  @param  {[selectedCanvas]} canvas a variable representing a
 *                             specific html5 canvas (optional)
 *  @param  {[String]} filename
 *  @param  {[String]} extension 'jpg' or 'png'
 *  @example
 *  <div class='norender'><code>
 *  function setup() {
 *    var c = createCanvas(100, 100);
 *    background(255, 0, 0);
 *    saveCanvas(c, 'myCanvas', 'jpg');
 *  }
 *  </code></div>
 *  <div class='norender'><code>
 *  // note that this example has the same result as above
 *  // if no canvas is specified, defaults to main canvas
 *  function setup() {
 *    createCanvas(100, 100);
 *    background(255, 0, 0);
 *    saveCanvas('myCanvas', 'jpg');
 *  }
 *  </code></div>
 *  <div class='norender'><code>
 *  // all of the following are valid
 *  saveCanvas(c, 'myCanvas', 'jpg');
 *  saveCanvas(c, 'myCanvas');
 *  saveCanvas(c);
 *  saveCanvas('myCanvas', 'png');
 *  saveCanvas('myCanvas');
 *  saveCanvas();
 *  </code></div>
 */
p5.prototype.saveCanvas = function() {

  var cnv, filename, extension;
  if (arguments.length === 3) {
    cnv = arguments[0];
    filename = arguments[1];
    extension = arguments[2];
  } else if (arguments.length === 2) {
    if (typeof arguments[0] === 'object') {
      cnv = arguments[0];
      filename = arguments[1];
    } else {
      filename = arguments[0];
      extension = arguments[1];
    }
  } else if (arguments.length === 1) {
    if (typeof arguments[0] === 'object') {
      cnv = arguments[0];
    } else {
      filename = arguments[0];
    }
  }

  if (cnv instanceof p5.Element) {
    cnv = cnv.elt;
  }
  if (!(cnv instanceof HTMLCanvasElement)) {
    cnv = null;
  }

  if (!extension) {
    extension = p5.prototype._checkFileExtension(filename, extension)[1];
    if (extension === '') {
      extension = 'png';
    }
  }

  if (!cnv) {
    if (this._curElement && this._curElement.elt) {
      cnv = this._curElement.elt;
    }
  }

  if ( p5.prototype._isSafari() ) {
    var aText = 'Hello, Safari user!\n';
    aText += 'Now capturing a screenshot...\n';
    aText += 'To save this image,\n';
    aText += 'go to File --> Save As.\n';
    alert(aText);
    window.location.href = cnv.toDataURL();
  } else {
    var mimeType;
    if (typeof(extension) === 'undefined') {
      extension = 'png';
      mimeType = 'image/png';
    }
    else {
      switch(extension){
      case 'png':
        mimeType = 'image/png';
        break;
      case 'jpeg':
        mimeType = 'image/jpeg';
        break;
      case 'jpg':
        mimeType = 'image/jpeg';
        break;
      default:
        mimeType = 'image/png';
        break;
      }
    }
    var downloadMime = 'image/octet-stream';
    var imageData = cnv.toDataURL(mimeType);
    imageData = imageData.replace(mimeType, downloadMime);

    p5.prototype.downloadFile(imageData, filename, extension);
  }
};

/**
 *  Capture a sequence of frames that can be used to create a movie.
 *  Accepts a callback. For example, you may wish to send the frames
 *  to a server where they can be stored or converted into a movie.
 *  If no callback is provided, the browser will pop up save dialogues in an
 *  attempt to download all of the images that have just been created. With the
 *  callback provided the image data isn't saved by default but instead passed
 *  as an argument to the callback function as an array of objects, with the
 *  size of array equal to the total number of frames.
 *
 *  @method saveFrames
 *  @param  {String}   filename
 *  @param  {String}   extension 'jpg' or 'png'
 *  @param  {Number}   duration  Duration in seconds to save the frames for.
 *  @param  {Number}   framerate  Framerate to save the frames in.
 *  @param  {Function} [callback] A callback function that will be executed
                                  to handle the image data. This function
                                  should accept an array as argument. The
                                  array will contain the specified number of
                                  frames of objects. Each object has three
                                  properties: imageData - an
                                  image/octet-stream, filename and extension.
 *  @example
 *  <div><code>
 *  function draw() {
 *    background(mouseX);
 *  }
 *
 *  function mousePressed() {
 *    saveFrames("out", "png", 1, 25, function(data){
 *      print(data);
 *    });
 *  }
 *  </code></div>
 */
p5.prototype.saveFrames = function(fName, ext, _duration, _fps, callback) {
  var duration = _duration || 3;
  duration = p5.prototype.constrain(duration, 0, 15);
  duration = duration * 1000;
  var fps = _fps || 15;
  fps = p5.prototype.constrain(fps, 0, 22);
  var count = 0;

  var makeFrame = p5.prototype._makeFrame;
  var cnv = this._curElement.elt;
  var frameFactory = setInterval(function(){
    makeFrame(fName + count, ext, cnv);
    count++;
  },1000/fps);

  setTimeout(function(){
    clearInterval(frameFactory);
    if (callback) {
      callback(frames);
    }
    else {
      for (var i = 0; i < frames.length; i++) {
        var f = frames[i];
        p5.prototype.downloadFile(f.imageData, f.filename, f.ext);
      }
    }
    frames = []; // clear frames
  }, duration + 0.01);
};

p5.prototype._makeFrame = function(filename, extension, _cnv) {
  var cnv;
  if (this) {
    cnv = this._curElement.elt;
  } else {
    cnv = _cnv;
  }
  var mimeType;
  if (!extension) {
    extension = 'png';
    mimeType = 'image/png';
  }
  else {
    switch(extension.toLowerCase()){
    case 'png':
      mimeType = 'image/png';
      break;
    case 'jpeg':
      mimeType = 'image/jpeg';
      break;
    case 'jpg':
      mimeType = 'image/jpeg';
      break;
    default:
      mimeType = 'image/png';
      break;
    }
  }
  var downloadMime = 'image/octet-stream';
  var imageData = cnv.toDataURL(mimeType);
  imageData = imageData.replace(mimeType, downloadMime);

  var thisFrame = {};
  thisFrame.imageData = imageData;
  thisFrame.filename = filename;
  thisFrame.ext = extension;
  frames.push(thisFrame);
};

module.exports = p5;

},{"../core/core":37}],56:[function(_dereq_,module,exports){
/**
 * @module Image
 * @submodule Loading & Displaying
 * @for p5
 * @requires core
 */

'use strict';

var p5 = _dereq_('../core/core');
var Filters = _dereq_('./filters');
var canvas = _dereq_('../core/canvas');
var constants = _dereq_('../core/constants');

_dereq_('../core/error_helpers');

/**
 * Loads an image from a path and creates a p5.Image from it.
 * <br><br>
 * The image may not be immediately available for rendering
 * If you want to ensure that the image is ready before doing
 * anything with it, place the loadImage() call in preload().
 * You may also supply a callback function to handle the image when it's ready.
 * <br><br>
 * The path to the image should be relative to the HTML file
 * that links in your sketch. Loading an from a URL or other
 * remote location may be blocked due to your browser's built-in
 * security.
 *
 * @method loadImage
 * @param  {String} path Path of the image to be loaded
 * @param  {Function(p5.Image)} [successCallback] Function to be called once
 *                                the image is loaded. Will be passed the
 *                                p5.Image.
 * @param  {Function(Event)}    [failureCallback] called with event error if
 *                                the image fails to load.
 * @return {p5.Image}             the p5.Image object
 * @example
 * <div>
 * <code>
 * var img;
 * function preload() {
 *   img = loadImage("assets/laDefense.jpg");
 * }
 * function setup() {
 *   image(img, 0, 0);
 * }
 * </code>
 * </div>
 * <div>
 * <code>
 * function setup() {
 *   // here we use a callback to display the image after loading
 *   loadImage("assets/laDefense.jpg", function(img) {
 *     image(img, 0, 0);
 *   });
 * }
 * </code>
 * </div>
 */
p5.prototype.loadImage = function(path, successCallback, failureCallback) {
  var img = new Image();
  var pImg = new p5.Image(1, 1, this);
  var decrementPreload = p5._getDecrementPreload.apply(this, arguments);

  img.onload = function() {
    pImg.width = pImg.canvas.width = img.width;
    pImg.height = pImg.canvas.height = img.height;

    // Draw the image into the backing canvas of the p5.Image
    pImg.drawingContext.drawImage(img, 0, 0);

    if (typeof successCallback === 'function') {
      successCallback(pImg);
    }
    if (decrementPreload && (successCallback !== decrementPreload)) {
      decrementPreload();
    }
  };
  img.onerror = function(e) {
    p5._friendlyFileLoadError(0,img.src);
    // don't get failure callback mixed up with decrementPreload
    if ((typeof failureCallback === 'function') &&
      (failureCallback !== decrementPreload)) {
      failureCallback(e);
    }
  };

  //set crossOrigin in case image is served which CORS headers
  //this will let us draw to canvas without tainting it.
  //see https://developer.mozilla.org/en-US/docs/HTML/CORS_Enabled_Image
  // When using data-uris the file will be loaded locally
  // so we don't need to worry about crossOrigin with base64 file types
  if(path.indexOf('data:image/') !== 0) {
    img.crossOrigin = 'Anonymous';
  }

  //start loading the image
  img.src = path;

  return pImg;
};

/**
 * Validates clipping params. Per drawImage spec sWidth and sHight cannot be
 * negative or greater than image intrinsic width and height
 * @private
 * @param {Number} sVal
 * @param {Number} iVal
 * @returns {Number}
 * @private
 */
function _sAssign(sVal, iVal) {
  if (sVal > 0 && sVal < iVal) {
    return sVal;
  }
  else {
    return iVal;
  }
}

/**
 * Draw an image to the main canvas of the p5js sketch
 *
 * @method image
 * @param  {p5.Image} img    the image to display
 * @param  {Number}   [sx=0]   The X coordinate of the top left corner of the
 *                             sub-rectangle of the source image to draw into
 *                             the destination canvas.
 * @param  {Number}   [sy=0]   The Y coordinate of the top left corner of the
 *                             sub-rectangle of the source image to draw into
 *                             the destination canvas.
 * @param {Number} [sWidth=img.width] The width of the sub-rectangle of the
 *                                    source image to draw into the destination
 *                                    canvas.
 * @param {Number} [sHeight=img.height] The height of the sub-rectangle of the
 *                                      source image to draw into the
 *                                      destination context.
 * @param  {Number}   [dx=0]    The X coordinate in the destination canvas at
 *                              which to place the top-left corner of the
 *                              source image.
 * @param  {Number}   [dy=0]    The Y coordinate in the destination canvas at
 *                              which to place the top-left corner of the
 *                              source image.
 * @param  {Number}   [dWidth]  The width to draw the image in the destination
 *                              canvas. This allows scaling of the drawn image.
 * @param  {Number}   [dHeight] The height to draw the image in the destination
 *                              canvas. This allows scaling of the drawn image.
 * @example
 * <div>
 * <code>
 * var img;
 * function preload() {
 *   img = loadImage("assets/laDefense.jpg");
 * }
 * function setup() {
 *   image(img, 0, 0);
 *   image(img, 0, 0, 100, 100);
 *   image(img, 0, 0, 100, 100, 0, 0, 100, 100);
 * }
 * </code>
 * </div>
 * <div>
 * <code>
 * function setup() {
 *   // here we use a callback to display the image after loading
 *   loadImage("assets/laDefense.jpg", function(img) {
 *     image(img, 0, 0);
 *   });
 * }
 * </code>
 * </div>
 */
p5.prototype.image =
  function(img, sx, sy, sWidth, sHeight, dx, dy, dWidth, dHeight) {
  // Temporarily disabling until options for p5.Graphics are added.
  // var args = new Array(arguments.length);
  // for (var i = 0; i < args.length; ++i) {
  //   args[i] = arguments[i];
  // }
  // this._validateParameters(
  //   'image',
  //   args,
  //   [
  //     ['p5.Image', 'Number', 'Number'],
  //     ['p5.Image', 'Number', 'Number', 'Number', 'Number']
  //   ]
  // );

  // set defaults per spec: https://goo.gl/3ykfOq
  if (arguments.length <= 5) {
    dx = sx || 0;
    dy = sy || 0;
    sx = 0;
    sy = 0;
    if (img.elt && img.elt.videoWidth && !img.canvas) { // video no canvas
      var actualW = img.elt.videoWidth;
      var actualH = img.elt.videoHeight;
      dWidth = sWidth || img.elt.width;
      dHeight = sHeight || img.elt.width*actualH/actualW;
      sWidth = actualW;
      sHeight = actualH;
    } else {
      dWidth = sWidth || img.width;
      dHeight = sHeight || img.height;
      sWidth = img.width;
      sHeight = img.height;
    }
  } else if (arguments.length === 9) {
    sx = sx || 0;
    sy = sy || 0;
    sWidth = _sAssign(sWidth, img.width);
    sHeight = _sAssign(sHeight, img.height);

    dx = dx || 0;
    dy = dy || 0;
    dWidth = dWidth || img.width;
    dHeight = dHeight || img.height;
  } else {
    throw 'Wrong number of arguments to image()';
  }

  var vals = canvas.modeAdjust(dx, dy, dWidth, dHeight,
    this._renderer._imageMode);

  // tint the image if there is a tint
  this._renderer.image(img, sx, sy, sWidth, sHeight, vals.x, vals.y, vals.w,
    vals.h);
};

/**
 * Sets the fill value for displaying images. Images can be tinted to
 * specified colors or made transparent by including an alpha value.
 * <br><br>
 * To apply transparency to an image without affecting its color, use
 * white as the tint color and specify an alpha value. For instance,
 * tint(255, 128) will make an image 50% transparent (assuming the default
 * alpha range of 0-255, which can be changed with colorMode()).
 * <br><br>
 * The value for the gray parameter must be less than or equal to the current
 * maximum value as specified by colorMode(). The default maximum value is
 * 255.
 *
 * @method tint
 * @param {Number|Array} v1   gray value, red or hue value (depending on the
 *                            current color mode), or color Array
 * @param {Number|Array} [v2] green or saturation value (depending on the
 *                            current color mode)
 * @param {Number|Array} [v3] blue or brightness value (depending on the
 *                            current color mode)
 * @param {Number|Array} [a]  opacity of the background
 * @example
 * <div>
 * <code>
 * var img;
 * function preload() {
 *   img = loadImage("assets/laDefense.jpg");
 * }
 * function setup() {
 *   image(img, 0, 0);
 *   tint(0, 153, 204);  // Tint blue
 *   image(img, 50, 0);
 * }
 * </code>
 * </div>
 *
 * <div>
 * <code>
 * var img;
 * function preload() {
 *   img = loadImage("assets/laDefense.jpg");
 * }
 * function setup() {
 *   image(img, 0, 0);
 *   tint(0, 153, 204, 126);  // Tint blue and set transparency
 *   image(img, 50, 0);
 * }
 * </code>
 * </div>
 *
 * <div>
 * <code>
 * var img;
 * function preload() {
 *   img = loadImage("assets/laDefense.jpg");
 * }
 * function setup() {
 *   image(img, 0, 0);
 *   tint(255, 126);  // Apply transparency without changing color
 *   image(img, 50, 0);
 * }
 * </code>
 * </div>
 */
p5.prototype.tint = function () {
  var c = this.color.apply(this, arguments);
  this._renderer._tint = c.levels;
};

/**
 * Removes the current fill value for displaying images and reverts to
 * displaying images with their original hues.
 *
 * @method noTint
 * @example
 * <div>
 * <code>
 * var img;
 * function preload() {
 *   img = loadImage("assets/bricks.jpg");
 * }
 * function setup() {
 *   tint(0, 153, 204);  // Tint blue
 *   image(img, 0, 0);
 *   noTint();  // Disable tint
 *   image(img, 50, 0);
 * }
 * </code>
 * </div>
 */
p5.prototype.noTint = function() {
  this._renderer._tint = null;
};

/**
 * Apply the current tint color to the input image, return the resulting
 * canvas.
 *
 * @param {p5.Image} The image to be tinted
 * @return {canvas} The resulting tinted canvas
 *
 */
p5.prototype._getTintedImageCanvas = function(img) {
  if (!img.canvas) {
    return img;
  }
  var pixels = Filters._toPixels(img.canvas);
  var tmpCanvas = document.createElement('canvas');
  tmpCanvas.width = img.canvas.width;
  tmpCanvas.height = img.canvas.height;
  var tmpCtx = tmpCanvas.getContext('2d');
  var id = tmpCtx.createImageData(img.canvas.width, img.canvas.height);
  var newPixels = id.data;

  for(var i = 0; i < pixels.length; i += 4) {
    var r = pixels[i];
    var g = pixels[i+1];
    var b = pixels[i+2];
    var a = pixels[i+3];

    newPixels[i] = r*this._renderer._tint[0]/255;
    newPixels[i+1] = g*this._renderer._tint[1]/255;
    newPixels[i+2] = b*this._renderer._tint[2]/255;
    newPixels[i+3] = a*this._renderer._tint[3]/255;
  }

  tmpCtx.putImageData(id, 0, 0);
  return tmpCanvas;
};

/**
 * Set image mode. Modifies the location from which images are drawn by
 * changing the way in which parameters given to image() are interpreted.
 * The default mode is imageMode(CORNER), which interprets the second and
 * third parameters of image() as the upper-left corner of the image. If
 * two additional parameters are specified, they are used to set the image's
 * width and height.
 * <br><br>
 * imageMode(CORNERS) interprets the second and third parameters of image()
 * as the location of one corner, and the fourth and fifth parameters as the
 * opposite corner.
 * <br><br>
 * imageMode(CENTER) interprets the second and third parameters of image()
 * as the image's center point. If two additional parameters are specified,
 * they are used to set the image's width and height.
 *
 * @method imageMode
 * @param {String} m The mode: either CORNER, CORNERS, or CENTER.
 * @example
 *
 * <div>
 * <code>
 * var img;
 * function preload() {
 *   img = loadImage("assets/bricks.jpg");
 * }
 * function setup() {
 *   imageMode(CORNER);
 *   image(img, 10, 10, 50, 50);
 * }
 * </code>
 * </div>
 *
 * <div>
 * <code>
 * var img;
 * function preload() {
 *   img = loadImage("assets/bricks.jpg");
 * }
 * function setup() {
 *   imageMode(CORNERS);
 *   image(img, 10, 10, 90, 40);
 * }
 * </code>
 * </div>
 *
 * <div>
 * <code>
 * var img;
 * function preload() {
 *   img = loadImage("assets/bricks.jpg");
 * }
 * function setup() {
 *   imageMode(CENTER);
 *   image(img, 50, 50, 80, 80);
 * }
 * </code>
 * </div>
 */
p5.prototype.imageMode = function(m) {
  if (m === constants.CORNER ||
    m === constants.CORNERS ||
    m === constants.CENTER) {
    this._renderer._imageMode = m;
  }
};


module.exports = p5;

},{"../core/canvas":35,"../core/constants":36,"../core/core":37,"../core/error_helpers":40,"./filters":54}],57:[function(_dereq_,module,exports){
/**
 * @module Image
 * @submodule Image
 * @requires core
 * @requires constants
 * @requires filters
 */

/**
 * This module defines the p5.Image class and P5 methods for
 * drawing images to the main display canvas.
 */

'use strict';

var p5 = _dereq_('../core/core');
var Filters = _dereq_('./filters');

/*
 * Class methods
 */

/**
 * Creates a new p5.Image. A p5.Image is a canvas backed representation of an
 * image.
 * <br><br>
 * p5 can display .gif, .jpg and .png images. Images may be displayed
 * in 2D and 3D space. Before an image is used, it must be loaded with the
 * loadImage() function. The p5.Image class contains fields for the width and
 * height of the image, as well as an array called pixels[] that contains the
 * values for every pixel in the image.
 * <br><br>
 * The methods described below allow easy access to the image's pixels and
 * alpha channel and simplify the process of compositing.
 * <br><br>
 * Before using the pixels[] array, be sure to use the loadPixels() method on
 * the image to make sure that the pixel data is properly loaded.
 *
 * @class p5.Image
 * @constructor
 * @param {Number} width
 * @param {Number} height
 * @param {Object} pInst An instance of a p5 sketch.
 */
p5.Image = function(width, height){
  /**
   * Image width.
   * @property width
   * @example
   * <div><code>
   * var img;
   * function preload() {
   *   img = loadImage("assets/rockies.jpg");
   * }
   *
   * function setup() {
   *   createCanvas(100, 100);
   *   image(img, 0, 0);
   *   for (var i=0; i < img.width; i++) {
   *     var c = img.get(i, img.height/2);
   *     stroke(c);
   *     line(i, height/2, i, height);
   *   }
   * }
   * </code></div>
   */
  this.width = width;
  /**
   * Image height.
   * @property height
   * @example
   * <div><code>
   * var img;
   * function preload() {
   *   img = loadImage("assets/rockies.jpg");
   * }
   *
   * function setup() {
   *   createCanvas(100, 100);
   *   image(img, 0, 0);
   *   for (var i=0; i < img.height; i++) {
   *     var c = img.get(img.width/2, i);
   *     stroke(c);
   *     line(0, i, width/2, i);
   *   }
   * }
   * </code></div>
   */
  this.height = height;
  this.canvas = document.createElement('canvas');
  this.canvas.width = this.width;
  this.canvas.height = this.height;
  this.drawingContext = this.canvas.getContext('2d');
  this._pixelDensity = 1;
  //used for webgl texturing only
  this.isTexture = false;
  /**
   * Array containing the values for all the pixels in the display window.
   * These values are numbers. This array is the size (include an appropriate
   * factor for pixelDensity) of the display window x4,
   * representing the R, G, B, A values in order for each pixel, moving from
   * left to right across each row, then down each column. Retina and other
   * high denisty displays may have more pixels[] (by a factor of
   * pixelDensity^2).
   * For example, if the image is 100x100 pixels, there will be 40,000. With
   * pixelDensity = 2, there will be 160,000. The first four values
   * (indices 0-3) in the array will be the R, G, B, A values of the pixel at
   * (0, 0). The second four values (indices 4-7) will contain the R, G, B, A
   * values of the pixel at (1, 0). More generally, to set values for a pixel
   * at (x, y):
   * <code><pre>var d = pixelDensity;
   * for (var i = 0; i < d; i++) {
   *   for (var j = 0; j < d; j++) {
   *     // loop over
   *     idx = 4*((y * d + j) * width * d + (x * d + i));
   *     pixels[idx] = r;
   *     pixels[idx+1] = g;
   *     pixels[idx+2] = b;
   *     pixels[idx+3] = a;
   *   }
   * }
   * </pre></code>
   * <br><br>
   * Before accessing this array, the data must loaded with the loadPixels()
   * function. After the array data has been modified, the updatePixels()
   * function must be run to update the changes.
   * @property pixels[]
   * @example
   * <div>
   * <code>
   * img = createImage(66, 66);
   * img.loadPixels();
   * for (i = 0; i < img.width; i++) {
   *   for (j = 0; j < img.height; j++) {
   *     img.set(i, j, color(0, 90, 102));
   *   }
   * }
   * img.updatePixels();
   * image(img, 17, 17);
   * </code>
   * </div>
   * <div>
   * <code>
   * var pink = color(255, 102, 204);
   * img = createImage(66, 66);
   * img.loadPixels();
   * for (var i = 0; i < 4*(width*height/2); i+=4) {
   *   img.pixels[i] = red(pink);
   *   img.pixels[i+1] = green(pink);
   *   img.pixels[i+2] = blue(pink);
   *   img.pixels[i+3] = alpha(pink);
   * }
   * img.updatePixels();
   * image(img, 17, 17);
   * </code>
   * </div>
   */
  this.pixels = [];
};

/**
 * Helper fxn for sharing pixel methods
 *
 */
p5.Image.prototype._setProperty = function (prop, value) {
  this[prop] = value;
};

/**
 * Loads the pixels data for this image into the [pixels] attribute.
 *
 * @method loadPixels
 * @example
 * <div><code>
 * var myImage;
 * var halfImage;
 *
 * function preload() {
 *   myImage = loadImage("assets/rockies.jpg");
 * }
 *
 * function setup() {
 *   myImage.loadPixels();
 *   halfImage = 4 * width * height/2;
 *   for(var i = 0; i < halfImage; i++){
 *     myImage.pixels[i+halfImage] = myImage.pixels[i];
 *   }
 *   myImage.updatePixels();
 * }
 *
 * function draw() {
 *   image(myImage, 0, 0);
 * }
 * </code></div>
 */
p5.Image.prototype.loadPixels = function(){
  p5.Renderer2D.prototype.loadPixels.call(this);
};

/**
 * Updates the backing canvas for this image with the contents of
 * the [pixels] array.
 *
 * @method updatePixels
 * @param {Integer|undefined} x x-offset of the target update area for the
 *                              underlying canvas
 * @param {Integer|undefined} y y-offset of the target update area for the
 *                              underlying canvas
 * @param {Integer|undefined} w height of the target update area for the
 *                              underlying canvas
 * @param {Integer|undefined} h height of the target update area for the
 *                              underlying canvas
 * @example
 * <div><code>
 * var myImage;
 * var halfImage;
 *
 * function preload() {
 *   myImage = loadImage("assets/rockies.jpg");
 * }
 *
 * function setup() {
 *   myImage.loadPixels();
 *   halfImage = 4 * width * height/2;
 *   for(var i = 0; i < halfImage; i++){
 *     myImage.pixels[i+halfImage] = myImage.pixels[i];
 *   }
 *   myImage.updatePixels();
 * }
 *
 * function draw() {
 *   image(myImage, 0, 0);
 * }
 * </code></div>
 */
p5.Image.prototype.updatePixels = function(x, y, w, h){
  p5.Renderer2D.prototype.updatePixels.call(this, x, y, w, h);
};

/**
 * Get a region of pixels from an image.
 *
 * If no params are passed, those whole image is returned,
 * if x and y are the only params passed a single pixel is extracted
 * if all params are passed a rectangle region is extracted and a p5.Image
 * is returned.
 *
 * Returns undefined if the region is outside the bounds of the image
 *
 * @method get
 * @param  {Number}               [x] x-coordinate of the pixel
 * @param  {Number}               [y] y-coordinate of the pixel
 * @param  {Number}               [w] width
 * @param  {Number}               [h] height
 * @return {Array/Color | p5.Image}     color of pixel at x,y in array format
 *                                    [R, G, B, A] or p5.Image
 * @example
 * <div><code>
 * var myImage;
 * var c;
 *
 * function preload() {
 *   myImage = loadImage("assets/rockies.jpg");
 * }
 *
 * function setup() {
 *   background(myImage);
 *   noStroke();
 *   c = myImage.get(60, 90);
 *   fill(c);
 *   rect(25, 25, 50, 50);
 * }
 *
 * //get() returns color here
 * </code></div>
 */
p5.Image.prototype.get = function(x, y, w, h){
  return p5.Renderer2D.prototype.get.call(this, x, y, w, h);
};

/**
 * Set the color of a single pixel or write an image into
 * this p5.Image.
 *
 * Note that for a large number of pixels this will
 * be slower than directly manipulating the pixels array
 * and then calling updatePixels().
 *
 * @method set
 * @param {Number}              x x-coordinate of the pixel
 * @param {Number}              y y-coordinate of the pixel
 * @param {Number|Array|Object}   a grayscale value | pixel array |
 *                                a p5.Color | image to copy
 * @example
 * <div>
 * <code>
 * img = createImage(66, 66);
 * img.loadPixels();
 * for (i = 0; i < img.width; i++) {
 *   for (j = 0; j < img.height; j++) {
 *     img.set(i, j, color(0, 90, 102, i % img.width * 2));
 *   }
 * }
 * img.updatePixels();
 * image(img, 17, 17);
 * image(img, 34, 34);
 * </code>
 * </div>
 */
p5.Image.prototype.set = function(x, y, imgOrCol){
  p5.Renderer2D.prototype.set.call(this, x, y, imgOrCol);
};

/**
 * Resize the image to a new width and height. To make the image scale
 * proportionally, use 0 as the value for the wide or high parameter.
 * For instance, to make the width of an image 150 pixels, and change
 * the height using the same proportion, use resize(150, 0).
 *
 * @method resize
 * @param {Number} width the resized image width
 * @param {Number} height the resized image height
 * @example
 * <div><code>
 * var img;
 *
 * function setup() {
 *   img = loadImage("assets/rockies.jpg");
 * }

 * function draw() {
 *   image(img, 0, 0);
 * }
 *
 * function mousePressed() {
 *   img.resize(50, 100);
 * }
 * </code></div>
 */
p5.Image.prototype.resize = function(width, height){

  // Copy contents to a temporary canvas, resize the original
  // and then copy back.
  //
  // There is a faster approach that involves just one copy and swapping the
  // this.canvas reference. We could switch to that approach if (as i think
  // is the case) there an expectation that the user would not hold a
  // reference to the backing canvas of a p5.Image. But since we do not
  // enforce that at the moment, I am leaving in the slower, but safer
  // implementation.

  // auto-resize
  if (width === 0 && height === 0) {
    width = this.canvas.width;
    height = this.canvas.height;
  } else if (width === 0) {
    width = this.canvas.width * height / this.canvas.height;
  } else if (height === 0) {
    height = this.canvas.height * width / this.canvas.width;
  }

  var tempCanvas = document.createElement('canvas');
  tempCanvas.width = width;
  tempCanvas.height = height;
  tempCanvas.getContext('2d').drawImage(this.canvas,
    0, 0, this.canvas.width, this.canvas.height,
    0, 0, tempCanvas.width, tempCanvas.height
  );


  // Resize the original canvas, which will clear its contents
  this.canvas.width = this.width = width;
  this.canvas.height = this.height = height;

  //Copy the image back

  this.drawingContext.drawImage(tempCanvas,
    0, 0, width, height,
    0, 0, width, height
  );

  if(this.pixels.length > 0){
    this.loadPixels();
  }
};

/**
 * Copies a region of pixels from one image to another. If no
 * srcImage is specified this is used as the source. If the source
 * and destination regions aren't the same size, it will
 * automatically resize source pixels to fit the specified
 * target region.
 *
 * @method copy
 * @param  {p5.Image|undefined} srcImage source image
 * @param  {Integer} sx X coordinate of the source's upper left corner
 * @param  {Integer} sy Y coordinate of the source's upper left corner
 * @param  {Integer} sw source image width
 * @param  {Integer} sh source image height
 * @param  {Integer} dx X coordinate of the destination's upper left corner
 * @param  {Integer} dy Y coordinate of the destination's upper left corner
 * @param  {Integer} dw destination image width
 * @param  {Integer} dh destination image height
 * @example
 * <div><code>
 * var photo;
 * var bricks;
 * var x;
 * var y;
 *
 * function preload() {
 *   photo = loadImage("assets/rockies.jpg");
 *   bricks = loadImage("assets/bricks.jpg");
 * }
 *
 * function setup() {
 *   x = bricks.width/2;
 *   y = bricks.height/2;
 *   photo.copy(bricks, 0, 0, x, y, 0, 0, x, y);
 *   image(photo, 0, 0);
 * }
 * </code></div>
 */
p5.Image.prototype.copy = function () {
  p5.prototype.copy.apply(this, arguments);
};

/**
 * Masks part of an image from displaying by loading another
 * image and using it's blue channel as an alpha channel for
 * this image.
 *
 * @method mask
 * @param {p5.Image} srcImage source image
 * @example
 * <div><code>
 * var photo, maskImage;
 * function preload() {
 *   photo = loadImage("assets/rockies.jpg");
 *   maskImage = loadImage("assets/mask2.png");
 * }
 *
 * function setup() {
 *   createCanvas(100, 100);
 *   photo.mask(maskImage);
 *   image(photo, 0, 0);
 * }
 * </code></div>
 *
 * http://blogs.adobe.com/webplatform/2013/01/28/blending-features-in-canvas/
 *
 */
// TODO: - Accept an array of alpha values.
//       - Use other channels of an image. p5 uses the
//       blue channel (which feels kind of arbitrary). Note: at the
//       moment this method does not match native processings original
//       functionality exactly.
p5.Image.prototype.mask = function(p5Image) {
  if(p5Image === undefined){
    p5Image = this;
  }
  var currBlend = this.drawingContext.globalCompositeOperation;

  var scaleFactor = 1;
  if (p5Image instanceof p5.Renderer) {
    scaleFactor = p5Image._pInst._pixelDensity;
  }

  var copyArgs = [
    p5Image,
    0,
    0,
    scaleFactor*p5Image.width,
    scaleFactor*p5Image.height,
    0,
    0,
    this.width,
    this.height
  ];

  this.drawingContext.globalCompositeOperation = 'destination-in';
  this.copy.apply(this, copyArgs);
  this.drawingContext.globalCompositeOperation = currBlend;
};

/**
 * Applies an image filter to a p5.Image
 *
 * @method filter
 * @param {String} operation one of threshold, gray, invert, posterize and
 *                           opaque see Filters.js for docs on each available
 *                           filter
 * @param {Number|undefined} value
 * @example
 * <div><code>
 * var photo1;
 * var photo2;
 *
 * function preload() {
 *   photo1 = loadImage("assets/rockies.jpg");
 *   photo2 = loadImage("assets/rockies.jpg");
 * }
 *
 * function setup() {
 *   photo2.filter("gray");
 *   image(photo1, 0, 0);
 *   image(photo2, width/2, 0);
 * }
 * </code></div>
 */
p5.Image.prototype.filter = function(operation, value) {
  Filters.apply(this.canvas, Filters[operation.toLowerCase()], value);
};

/**
 * Copies a region of pixels from one image to another, using a specified
 * blend mode to do the operation.
 *
 * @method blend
 * @param  {p5.Image|undefined} srcImage source image
 * @param  {Integer} sx X coordinate of the source's upper left corner
 * @param  {Integer} sy Y coordinate of the source's upper left corner
 * @param  {Integer} sw source image width
 * @param  {Integer} sh source image height
 * @param  {Integer} dx X coordinate of the destination's upper left corner
 * @param  {Integer} dy Y coordinate of the destination's upper left corner
 * @param  {Integer} dw destination image width
 * @param  {Integer} dh destination image height
 * @param  {Integer} blendMode the blend mode
 *
 * Available blend modes are: normal | multiply | screen | overlay |
 *            darken | lighten | color-dodge | color-burn | hard-light |
 *            soft-light | difference | exclusion | hue | saturation |
 *            color | luminosity
 *
 *
 * http://blogs.adobe.com/webplatform/2013/01/28/blending-features-in-canvas/
 * @example
 * <div><code>
 * var mountains;
 * var bricks;
 *
 * function preload() {
 *   mountains = loadImage("assets/rockies.jpg");
 *   bricks = loadImage("assets/bricks_third.jpg");
 * }
 *
 * function setup() {
 *   mountains.blend(bricks, 0, 0, 33, 100, 67, 0, 33, 100, ADD);
 *   image(mountains, 0, 0);
 *   image(bricks, 0, 0);
 * }
 * </code></div>
 * <div><code>
 * var mountains;
 * var bricks;
 *
 * function preload() {
 *   mountains = loadImage("assets/rockies.jpg");
 *   bricks = loadImage("assets/bricks_third.jpg");
 * }
 *
 * function setup() {
 *   mountains.blend(bricks, 0, 0, 33, 100, 67, 0, 33, 100, DARKEST);
 *   image(mountains, 0, 0);
 *   image(bricks, 0, 0);
 * }
 * </code></div>
 * <div><code>
 * var mountains;
 * var bricks;
 *
 * function preload() {
 *   mountains = loadImage("assets/rockies.jpg");
 *   bricks = loadImage("assets/bricks_third.jpg");
 * }
 *
 * function setup() {
 *   mountains.blend(bricks, 0, 0, 33, 100, 67, 0, 33, 100, LIGHTEST);
 *   image(mountains, 0, 0);
 *   image(bricks, 0, 0);
 * }
 * </code></div>
 */
p5.Image.prototype.blend = function() {
  p5.prototype.blend.apply(this, arguments);
};

/**
 * Saves the image to a file and force the browser to download it.
 * Accepts two strings for filename and file extension
 * Supports png (default) and jpg.
 *
 * @method save
 * @param {String} filename give your file a name
 * @param  {String} extension 'png' or 'jpg'
 * @example
 * <div><code>
 * var photo;
 *
 * function preload() {
 *   photo = loadImage("assets/rockies.jpg");
 * }
 *
 * function draw() {
 *   image(photo, 0, 0);
 * }
 *
 * function keyTyped() {
 *   if (key == 's') {
 *     photo.save("photo", "png");
 *   }
 * }
 * </code></div>
 */
p5.Image.prototype.save = function(filename, extension) {
  var mimeType;
  if (!extension) {
    extension = 'png';
    mimeType = 'image/png';
  }
  else {
    // en.wikipedia.org/wiki/Comparison_of_web_browsers#Image_format_support
    switch(extension.toLowerCase()){
    case 'png':
      mimeType = 'image/png';
      break;
    case 'jpeg':
      mimeType = 'image/jpeg';
      break;
    case 'jpg':
      mimeType = 'image/jpeg';
      break;
    default:
      mimeType = 'image/png';
      break;
    }
  }
  var downloadMime = 'image/octet-stream';
  var imageData = this.canvas.toDataURL(mimeType);
  imageData = imageData.replace(mimeType, downloadMime);

  //Make the browser download the file
  p5.prototype.downloadFile(imageData, filename, extension);
};

module.exports = p5.Image;
},{"../core/core":37,"./filters":54}],58:[function(_dereq_,module,exports){
/**
 * @module Image
 * @submodule Pixels
 * @for p5
 * @requires core
 */

'use strict';

var p5 = _dereq_('../core/core');
var Filters = _dereq_('./filters');
_dereq_('../color/p5.Color');

/**
 * <a href='https://developer.mozilla.org/en-US/docs/Web/JavaScript/Reference
 * /Global_Objects/Uint8ClampedArray' target='_blank'>Uint8ClampedArray</a>
 * containing the values for all the pixels in the display window.
 * These values are numbers. This array is the size (include an appropriate
 * factor for pixelDensity) of the display window x4,
 * representing the R, G, B, A values in order for each pixel, moving from
 * left to right across each row, then down each column. Retina and other
 * high denisty displays will have more pixels[] (by a factor of
 * pixelDensity^2).
 * For example, if the image is 100x100 pixels, there will be 40,000. On a
 * retina display, there will be 160,000.
 * <br><br>
 * The first four values (indices 0-3) in the array will be the R, G, B, A
 * values of the pixel at (0, 0). The second four values (indices 4-7) will
 * contain the R, G, B, A values of the pixel at (1, 0). More generally, to
 * set values for a pixel at (x, y):
 * <code><pre>
 * var d = pixelDensity;
 * for (var i = 0; i < d; i++) {
 *   for (var j = 0; j < d; j++) {
 *     // loop over
 *     idx = 4 * ((y * d + j) * width * d + (x * d + i));
 *     pixels[idx] = r;
 *     pixels[idx+1] = g;
 *     pixels[idx+2] = b;
 *     pixels[idx+3] = a;
 *   }
 * }
 * </pre></code>
 *
 * <p>While the above method is complex, it is flexible enough to work with
 * any pixelDensity. Note that set() will automatically take care of
 * setting all the appropriate values in pixels[] for a given (x, y) at
 * any pixelDensity, but the performance may not be as fast when lots of
 * modifications are made to the pixel array.
 * <br><br>
 * Before accessing this array, the data must loaded with the loadPixels()
 * function. After the array data has been modified, the updatePixels()
 * function must be run to update the changes.
 * <br><br>
 * Note that this is not a standard javascript array.  This means that
 * standard javascript functions such as <code>slice()</code> or
 * <code>arrayCopy()</code> do not
 * work.</p>
 *
 * @property pixels[]
 * @example
 * <div>
 * <code>
 * var pink = color(255, 102, 204);
 * loadPixels();
 * var d = pixelDensity();
 * var halfImage = 4 * (width * d) * (height/2 * d);
 * for (var i = 0; i < halfImage; i+=4) {
 *   pixels[i] = red(pink);
 *   pixels[i+1] = green(pink);
 *   pixels[i+2] = blue(pink);
 *   pixels[i+3] = alpha(pink);
 * }
 * updatePixels();
 * </code>
 * </div>
 */
p5.prototype.pixels = [];

/**
 * Copies a region of pixels from one image to another, using a specified
 * blend mode to do the operation.<br><br>
 * Available blend modes are: BLEND | DARKEST | LIGHTEST | DIFFERENCE |
 * MULTIPLY| EXCLUSION | SCREEN | REPLACE | OVERLAY | HARD_LIGHT |
 * SOFT_LIGHT | DODGE | BURN | ADD | NORMAL
 *
 *
 * @method blend
 * @param  {p5.Image|undefined} srcImage source image
 * @param  {Integer} sx X coordinate of the source's upper left corner
 * @param  {Integer} sy Y coordinate of the source's upper left corner
 * @param  {Integer} sw source image width
 * @param  {Integer} sh source image height
 * @param  {Integer} dx X coordinate of the destination's upper left corner
 * @param  {Integer} dy Y coordinate of the destination's upper left corner
 * @param  {Integer} dw destination image width
 * @param  {Integer} dh destination image height
 * @param  {Integer} blendMode the blend mode
 *
 * @example
 * <div><code>
 * var img0;
 * var img1;
 *
 * function preload() {
 *   img0 = loadImage("assets/rockies.jpg");
 *   img1 = loadImage("assets/bricks_third.jpg");
 * }
 *
 * function setup() {
 *   background(img0);
 *   image(img1, 0, 0);
 *   blend(img1, 0, 0, 33, 100, 67, 0, 33, 100, LIGHTEST);
 * }
 * </code></div>
 * <div><code>
 * var img0;
 * var img1;
 *
 * function preload() {
 *   img0 = loadImage("assets/rockies.jpg");
 *   img1 = loadImage("assets/bricks_third.jpg");
 * }
 *
 * function setup() {
 *   background(img0);
 *   image(img1, 0, 0);
 *   blend(img1, 0, 0, 33, 100, 67, 0, 33, 100, DARKEST);
 * }
 * </code></div>
 * <div><code>
 * var img0;
 * var img1;
 *
 * function preload() {
 *   img0 = loadImage("assets/rockies.jpg");
 *   img1 = loadImage("assets/bricks_third.jpg");
 * }
 *
 * function setup() {
 *   background(img0);
 *   image(img1, 0, 0);
 *   blend(img1, 0, 0, 33, 100, 67, 0, 33, 100, ADD);
 * }
 * </code></div>
 */
p5.prototype.blend = function() {
  if (this._renderer) {
    this._renderer.blend.apply(this._renderer, arguments);
  } else {
    p5.Renderer2D.prototype.blend.apply(this, arguments);
  }
};

/**
 * Copies a region of the canvas to another region of the canvas
 * and copies a region of pixels from an image used as the srcImg parameter
 * into the canvas srcImage is specified this is used as the source. If
 * the source and destination regions aren't the same size, it will
 * automatically resize source pixels to fit the specified
 * target region.
 *
 * @method copy
 * @param  {p5.Image|undefined} srcImage source image
 * @param  {Integer} sx X coordinate of the source's upper left corner
 * @param  {Integer} sy Y coordinate of the source's upper left corner
 * @param  {Integer} sw source image width
 * @param  {Integer} sh source image height
 * @param  {Integer} dx X coordinate of the destination's upper left corner
 * @param  {Integer} dy Y coordinate of the destination's upper left corner
 * @param  {Integer} dw destination image width
 * @param  {Integer} dh destination image height
 *
 * @example
 * <div><code>
 * var img;
 *
 * function preload() {
 *   img = loadImage("assets/rockies.jpg");
 * }
 *
 * function setup() {
 *   background(img);
 *   copy(img, 7, 22, 10, 10, 35, 25, 50, 50);
 *   stroke(255);
 *   noFill();
 *   // Rectangle shows area being copied
 *   rect(7, 22, 10, 10);
 * }
 * </code></div>
 */
p5.prototype.copy = function () {
  p5.Renderer2D._copyHelper.apply(this, arguments);
};

/**
 * Applies a filter to the canvas.
 * <br><br>
 *
 * The presets options are:
 * <br><br>
 *
 * THRESHOLD
 * Converts the image to black and white pixels depending if they are above or
 * below the threshold defined by the level parameter. The parameter must be
 * between 0.0 (black) and 1.0 (white). If no level is specified, 0.5 is used.
 * <br><br>
 *
 * GRAY
 * Converts any colors in the image to grayscale equivalents. No parameter
 * is used.
 * <br><br>
 *
 * OPAQUE
 * Sets the alpha channel to entirely opaque. No parameter is used.
 * <br><br>
 *
 * INVERT
 * Sets each pixel to its inverse value. No parameter is used.
 * <br><br>
 *
 * POSTERIZE
 * Limits each channel of the image to the number of colors specified as the
 * parameter. The parameter can be set to values between 2 and 255, but
 * results are most noticeable in the lower ranges.
 * <br><br>
 *
 * BLUR
 * Executes a Guassian blur with the level parameter specifying the extent
 * of the blurring. If no parameter is used, the blur is equivalent to
 * Guassian blur of radius 1. Larger values increase the blur.
 * <br><br>
 *
 * ERODE
 * Reduces the light areas. No parameter is used.
 * <br><br>
 *
 * DILATE
 * Increases the light areas. No parameter is used.
 *
 * @method filter
 * @param  {String} filterType
 * @param  {Number} filterParam an optional parameter unique
 *  to each filter, see above
 *
 *
 * @example
 * <div>
 * <code>
 * var img;
 * function preload() {
 *   img = loadImage("assets/bricks.jpg");
 * }
 * function setup() {
 *  image(img, 0, 0);
 *  filter(THRESHOLD);
 * }
 * </code>
 * </div>
 *
 * <div>
 * <code>
 * var img;
 * function preload() {
 *   img = loadImage("assets/bricks.jpg");
 * }
 * function setup() {
 *  image(img, 0, 0);
 *  filter(GRAY);
 * }
 * </code>
 * </div>
 *
 * <div>
 * <code>
 * var img;
 * function preload() {
 *   img = loadImage("assets/bricks.jpg");
 * }
 * function setup() {
 *  image(img, 0, 0);
 *  filter(OPAQUE);
 * }
 * </code>
 * </div>
 *
 * <div>
 * <code>
 * var img;
 * function preload() {
 *   img = loadImage("assets/bricks.jpg");
 * }
 * function setup() {
 *  image(img, 0, 0);
 *  filter(INVERT);
 * }
 * </code>
 * </div>
 *
 * <div>
 * <code>
 * var img;
 * function preload() {
 *   img = loadImage("assets/bricks.jpg");
 * }
 * function setup() {
 *  image(img, 0, 0);
 *  filter(POSTERIZE,3);
 * }
 * </code>
 * </div>
 *
 * <div>
 * <code>
 * var img;
 * function preload() {
 *   img = loadImage("assets/bricks.jpg");
 * }
 * function setup() {
 *  image(img, 0, 0);
 *  filter(DILATE);
 * }
 * </code>
 * </div>
 *
 * <div>
 * <code>
 * var img;
 * function preload() {
 *   img = loadImage("assets/bricks.jpg");
 * }
 * function setup() {
 *  image(img, 0, 0);
 *  filter(BLUR,3);
 * }
 * </code>
 * </div>
 *
 * <div>
 * <code>
 * var img;
 * function preload() {
 *   img = loadImage("assets/bricks.jpg");
 * }
 * function setup() {
 *  image(img, 0, 0);
 *  filter(ERODE);
 * }
 * </code>
 * </div>
 */
p5.prototype.filter = function(operation, value) {
  Filters.apply(this.canvas, Filters[operation.toLowerCase()], value);
};

/**
 * Returns an array of [R,G,B,A] values for any pixel or grabs a section of
 * an image. If no parameters are specified, the entire image is returned.
 * Use the x and y parameters to get the value of one pixel. Get a section of
 * the display window by specifying additional w and h parameters. When
 * getting an image, the x and y parameters define the coordinates for the
 * upper-left corner of the image, regardless of the current imageMode().
 * <br><br>
 * If the pixel requested is outside of the image window, [0,0,0,255] is
 * returned. To get the numbers scaled according to the current color ranges
 * and taking into account colorMode, use getColor instead of get.
 * <br><br>
 * Getting the color of a single pixel with get(x, y) is easy, but not as fast
 * as grabbing the data directly from pixels[]. The equivalent statement to
 * get(x, y) using pixels[] with pixel density d is
 * <code>
 * var off = (y * width + x) * d * 4;
 * [pixels[off],
 * pixels[off+1],
 * pixels[off+2],
 * pixels[off+3]]</code>
 * <br><br>
 * See the reference for pixels[] for more information.
 *
 * @method get
 * @param  {Number}         [x] x-coordinate of the pixel
 * @param  {Number}         [y] y-coordinate of the pixel
 * @param  {Number}         [w] width
 * @param  {Number}         [h] height
 * @return {Array|p5.Image}     values of pixel at x,y in array format
 *                              [R, G, B, A] or p5.Image
 * @example
 * <div>
 * <code>
 * var img;
 * function preload() {
 *   img = loadImage("assets/rockies.jpg");
 * }
 * function setup() {
 *   image(img, 0, 0);
 *   var c = get();
 *   image(c, width/2, 0);
 * }
 * </code>
 * </div>
 *
 * <div>
 * <code>
 * var img;
 * function preload() {
 *   img = loadImage("assets/rockies.jpg");
 * }
 * function setup() {
 *   image(img, 0, 0);
 *   var c = get(50, 90);
 *   fill(c);
 *   noStroke();
 *   rect(25, 25, 50, 50);
 * }
 * </code>
 * </div>
 */
p5.prototype.get = function(x, y, w, h){
  return this._renderer.get(x, y, w, h);
};

/**
 * Loads the pixel data for the display window into the pixels[] array. This
 * function must always be called before reading from or writing to pixels[].
 *
 * @method loadPixels
 * @example
 * <div>
 * <code>
 * var img;
 * function preload() {
 *   img = loadImage("assets/rockies.jpg");
 * }
 *
 * function setup() {
 *   image(img, 0, 0);
 *   var d = pixelDensity();
 *   var halfImage = 4 * (img.width * d) *
       (img.height/2 * d);
 *   loadPixels();
 *   for (var i = 0; i < halfImage; i++) {
 *     pixels[i+halfImage] = pixels[i];
 *   }
 *   updatePixels();
 * }
 * </code>
 * </div>
 */
p5.prototype.loadPixels = function() {
  this._renderer.loadPixels();
};

/**
 * <p>Changes the color of any pixel, or writes an image directly to the
 * display window.</p>
 * <p>The x and y parameters specify the pixel to change and the c parameter
 * specifies the color value. This can be a p5.Color object, or [R, G, B, A]
 * pixel array. It can also be a single grayscale value.
 * When setting an image, the x and y parameters define the coordinates for
 * the upper-left corner of the image, regardless of the current imageMode().
 * </p>
 * <p>
 * After using set(), you must call updatePixels() for your changes to
 * appear.  This should be called once all pixels have been set.
 * </p>
 * <p>Setting the color of a single pixel with set(x, y) is easy, but not as
 * fast as putting the data directly into pixels[]. Setting the pixels[]
 * values directly may be complicated when working with a retina display,
 * but will perform better when lots of pixels need to be set directly on
 * every loop.</p>
 * <p>See the reference for pixels[] for more information.</p>
 *
 * @method set
 * @param {Number}              x x-coordinate of the pixel
 * @param {Number}              y y-coordinate of the pixel
 * @param {Number|Array|Object} c insert a grayscale value | a pixel array |
 *                                a p5.Color object | a p5.Image to copy
 * @example
 * <div>
 * <code>
 * var black = color(0);
 * set(30, 20, black);
 * set(85, 20, black);
 * set(85, 75, black);
 * set(30, 75, black);
 * updatePixels();
 * </code>
 * </div>
 *
 * <div>
 * <code>
 * for (var i = 30; i < width-15; i++) {
 *   for (var j = 20; j < height-25; j++) {
 *     var c = color(204-j, 153-i, 0);
 *     set(i, j, c);
 *   }
 * }
 * updatePixels();
 * </code>
 * </div>
 *
 * <div>
 * <code>
 * var img;
 * function preload() {
 *   img = loadImage("assets/rockies.jpg");
 * }
 *
 * function setup() {
 *   set(0, 0, img);
 *   updatePixels();
 *   line(0, 0, width, height);
 *   line(0, height, width, 0);
 * }
 * </code>
 * </div>
 */
p5.prototype.set = function (x, y, imgOrCol) {
  this._renderer.set(x, y, imgOrCol);
};
/**
 * Updates the display window with the data in the pixels[] array.
 * Use in conjunction with loadPixels(). If you're only reading pixels from
 * the array, there's no need to call updatePixels() — updating is only
 * necessary to apply changes. updatePixels() should be called anytime the
 * pixels array is manipulated or set() is called.
 *
 * @method updatePixels
 * @param  {Number} [x]    x-coordinate of the upper-left corner of region
 *                         to update
 * @param  {Number} [y]    y-coordinate of the upper-left corner of region
 *                         to update
 * @param  {Number} [w]    width of region to update
 * @param  {Number} [w]    height of region to update
 * @example
 * <div>
 * <code>
 * var img;
 * function preload() {
 *   img = loadImage("assets/rockies.jpg");
 * }
 *
 * function setup() {
 *   image(img, 0, 0);
 *   var halfImage = 4 * (img.width * pixelDensity()) *
 *     (img.height * pixelDensity()/2);
 *   loadPixels();
 *   for (var i = 0; i < halfImage; i++) {
 *     pixels[i+halfImage] = pixels[i];
 *   }
 *   updatePixels();
 * }
 * </code>
 * </div>
 */
p5.prototype.updatePixels = function (x, y, w, h) {
  // graceful fail - if loadPixels() or set() has not been called, pixel
  // array will be empty, ignore call to updatePixels()
  if (this.pixels.length === 0) {
    return;
  }
  this._renderer.updatePixels(x, y, w, h);
};

module.exports = p5;

},{"../color/p5.Color":31,"../core/core":37,"./filters":54}],59:[function(_dereq_,module,exports){
/**
 * @module IO
 * @submodule Input
 * @for p5
 * @requires core
 * @requires reqwest
 */

'use strict';

var p5 = _dereq_('../core/core');
var reqwest = _dereq_('reqwest');
var opentype = _dereq_('opentype.js');
_dereq_('../core/error_helpers');

/**
 * Checks if we are in preload and returns the last arg which will be the
 * _decrementPreload function if called from a loadX() function.  Should
 * only be used in loadX() functions.
 * @private
 */
p5._getDecrementPreload = function () {
  var decrementPreload = arguments[arguments.length - 1];

  // when in preload decrementPreload will always be the last arg as it is set
  // with args.push() before invocation in _wrapPreload
  if ((window.preload || (this && this.preload)) &&
    typeof decrementPreload === 'function') {
    return decrementPreload;
  } else {
    return null;
  }
};

/**
 * Loads an opentype font file (.otf, .ttf) from a file or a URL,
 * and returns a PFont Object. This method is asynchronous,
 * meaning it may not finish before the next line in your sketch
 * is executed.
 * <br><br>
 * The path to the font should be relative to the HTML file
 * that links in your sketch. Loading an from a URL or other
 * remote location may be blocked due to your browser's built-in
 * security.
 *
 * @method loadFont
 * @param  {String}        path       name of the file or url to load
 * @param  {Function}      [callback] function to be executed after
 *                                    loadFont()
 *                                    completes
 * @return {Object}                   p5.Font object
 * @example
 *
 * <p>Calling loadFont() inside preload() guarantees that the load
 * operation will have completed before setup() and draw() are called.</p>
 *
 * <div><code>
 * var myFont;
 * function preload() {
 *   myFont = loadFont('assets/AvenirNextLTPro-Demi.otf');
 * }
 *
 * function setup() {
 *   fill('#ED225D');
 *   textFont(myFont);
 *   textSize(36);
 *   text('p5*js', 10, 50);
 * }
 * </code></div>
 *
 * Outside of preload(), you may supply a callback function to handle the
 * object:
 *
 * <div><code>
 * function setup() {
 *   loadFont('assets/AvenirNextLTPro-Demi.otf', drawText);
 * }
 *
 * function drawText(font) {
 *   fill('#ED225D');
 *   textFont(font, 36);
 *   text('p5*js', 10, 50);
 * }
 *
 * </code></div>
 *
 * <p>You can also use the string name of the font to style other HTML
 * elements.</p>
 *
 * <div><code>
 * var myFont;
 *
 * function preload() {
 *   myFont = loadFont('assets/Avenir.otf');
 * }
 *
 * function setup() {
 *   var myDiv = createDiv('hello there');
 *   myDiv.style('font-family', 'Avenir');
 * }
 * </code></div>
 */
p5.prototype.loadFont = function (path, onSuccess, onError) {

  var p5Font = new p5.Font(this);
  var decrementPreload = p5._getDecrementPreload.apply(this, arguments);

  opentype.load(path, function (err, font) {

    if (err) {

      if ((typeof onError !== 'undefined') && (onError !== decrementPreload)) {
        return onError(err);
      }
      throw err;
    }

    p5Font.font = font;

    if (typeof onSuccess !== 'undefined') {
      onSuccess(p5Font);
    }

    if (decrementPreload && (onSuccess !== decrementPreload)) {
      decrementPreload();
    }

    // check that we have an acceptable font type
    var validFontTypes = [ 'ttf', 'otf', 'woff', 'woff2' ],
      fileNoPath = path.split('\\').pop().split('/').pop(),
      lastDotIdx = fileNoPath.lastIndexOf('.'), fontFamily, newStyle,
      fileExt = lastDotIdx < 1 ? null : fileNoPath.substr(lastDotIdx + 1);

    // if so, add it to the DOM (name-only) for use with p5.dom
    if (validFontTypes.indexOf(fileExt) > -1) {

      fontFamily = fileNoPath.substr(0, lastDotIdx);
      newStyle = document.createElement('style');
      newStyle.appendChild(document.createTextNode('\n@font-face {' +
        '\nfont-family: ' + fontFamily + ';\nsrc: url(' + path + ');\n}\n'));
      document.head.appendChild(newStyle);
    }

  });

  return p5Font;
};

//BufferedReader
p5.prototype.createInput = function () {
  // TODO
  throw 'not yet implemented';
};

p5.prototype.createReader = function () {
  // TODO
  throw 'not yet implemented';
};

p5.prototype.loadBytes = function () {
  // TODO
  throw 'not yet implemented';
};

/**
 * Loads a JSON file from a file or a URL, and returns an Object or Array.
 * This method is asynchronous, meaning it may not finish before the next
 * line in your sketch is executed.
 *
 * @method loadJSON
 * @param  {String}        path       name of the file or url to load
 * @param  {Function}      [callback] function to be executed after
 *                                    loadJSON() completes, data is passed
 *                                    in as first argument
 * @param  {Function}      [errorCallback] function to be executed if
 *                                    there is an error, response is passed
 *                                    in as first argument
 * @param  {String}        [datatype] "json" or "jsonp"
 * @return {Object|Array}             JSON data
 * @example
 *
 * <p>Calling loadJSON() inside preload() guarantees to complete the
 * operation before setup() and draw() are called.</p>
 *
 * <div><code>
 * var weather;
 * function preload() {
 *   var url = 'http://api.openweathermap.org/data/2.5/weather?q=London,UK'+
 *    '&APPID=7bbbb47522848e8b9c26ba35c226c734';
 *   weather = loadJSON(url);
 * }
 *
 * function setup() {
 *   noLoop();
 * }
 *
 * function draw() {
 *   background(200);
 *   // get the humidity value out of the loaded JSON
 *   var humidity = weather.main.humidity;
 *   fill(0, humidity); // use the humidity value to set the alpha
 *   ellipse(width/2, height/2, 50, 50);
 * }
 * </code></div>
 *
 *
 * <p>Outside of preload(), you may supply a callback function to handle the
 * object:</p>
 * <div><code>
 * function setup() {
 *   noLoop();
 *   var url = 'http://api.openweathermap.org/data/2.5/weather?q=NewYork'+
 *    '&APPID=7bbbb47522848e8b9c26ba35c226c734';
 *   loadJSON(url, drawWeather);
 * }
 *
 * function draw() {
 *   background(200);
 * }
 *
 * function drawWeather(weather) {
 *   // get the humidity value out of the loaded JSON
 *   var humidity = weather.main.humidity;
 *   fill(0, humidity); // use the humidity value to set the alpha
 *   ellipse(width/2, height/2, 50, 50);
 * }
 * </code></div>
 *
 */
p5.prototype.loadJSON = function () {
  var path = arguments[0];
  var callback = arguments[1];
  var errorCallback;
  var decrementPreload = p5._getDecrementPreload.apply(this, arguments);

  var ret = []; // array needed for preload
  // assume jsonp for URLs
  var t = 'json'; //= path.indexOf('http') === -1 ? 'json' : 'jsonp';

  // check for explicit data type argument
  for (var i = 2; i < arguments.length; i++) {
    var arg = arguments[i];
    if (typeof arg === 'string') {
      if (arg === 'jsonp' || arg === 'json') {
        t = arg;
      }
    } else if (typeof arg === 'function') {
      errorCallback = arg;
    }
  }

  reqwest({
    url: path,
    type: t,
    crossOrigin: true,
    error: function (resp) {
      // pass to error callback if defined
      if (errorCallback) {
        errorCallback(resp);
      } else { // otherwise log error msg
        console.log(resp.statusText);
      }
    },
    success: function (resp) {
      for (var k in resp) {
        ret[k] = resp[k];
      }
      if (typeof callback !== 'undefined') {
        callback(resp);
      }
      if (decrementPreload && (callback !== decrementPreload)) {
        decrementPreload();
      }
    }
  });

  return ret;
};

/**
 * Reads the contents of a file and creates a String array of its individual
 * lines. If the name of the file is used as the parameter, as in the above
 * example, the file must be located in the sketch directory/folder.
 * <br><br>
 * Alternatively, the file maybe be loaded from anywhere on the local
 * computer using an absolute path (something that starts with / on Unix and
 * Linux, or a drive letter on Windows), or the filename parameter can be a
 * URL for a file found on a network.
 * <br><br>
 * This method is asynchronous, meaning it may not finish before the next
 * line in your sketch is executed.
 *
 * @method loadStrings
 * @param  {String}   filename   name of the file or url to load
 * @param  {Function} [callback] function to be executed after loadStrings()
 *                               completes, Array is passed in as first
 *                               argument
 * @param  {Function} [errorCallback] function to be executed if
 *                               there is an error, response is passed
 *                               in as first argument
 * @return {Array}               Array of Strings
 * @example
 *
 * <p>Calling loadStrings() inside preload() guarantees to complete the
 * operation before setup() and draw() are called.</p>
 *
 * <div><code>
 * var result;
 * function preload() {
 *   result = loadStrings('assets/test.txt');
 * }

 * function setup() {
 *   background(200);
 *   var ind = floor(random(result.length));
 *   text(result[ind], 10, 10, 80, 80);
 * }
 * </code></div>
 *
 * <p>Outside of preload(), you may supply a callback function to handle the
 * object:</p>
 *
 * <div><code>
 * function setup() {
 *   loadStrings('assets/test.txt', pickString);
 * }
 *
 * function pickString(result) {
 *   background(200);
 *   var ind = floor(random(result.length));
 *   text(result[ind], 10, 10, 80, 80);
 * }
 * </code></div>
 */
p5.prototype.loadStrings = function (path, callback, errorCallback) {
  var ret = [];
  var req = new XMLHttpRequest();
  var decrementPreload = p5._getDecrementPreload.apply(this, arguments);

  req.addEventListener('error', function (resp) {
    if (errorCallback) {
      errorCallback(resp);
    } else {
      console.log(resp.responseText);
    }
  });

  req.open('GET', path, true);
  req.onreadystatechange = function () {
    if (req.readyState === 4) {
      if (req.status === 200) {
        var arr = req.responseText.match(/[^\r\n]+/g);
        for (var k in arr) {
          ret[k] = arr[k];
        }
        if (typeof callback !== 'undefined') {
          callback(ret);
        }
        if (decrementPreload && (callback !== decrementPreload)) {
          decrementPreload();
        }
      } else {
        if (errorCallback) {
          errorCallback(req);
        } else {
          console.log(req.statusText);
        }
        //p5._friendlyFileLoadError(3, path);
      }
    }
  };
  req.send(null);
  return ret;
};

/**
 * <p>Reads the contents of a file or URL and creates a p5.Table object with
 * its values. If a file is specified, it must be located in the sketch's
 * "data" folder. The filename parameter can also be a URL to a file found
 * online. By default, the file is assumed to be comma-separated (in CSV
 * format). Table only looks for a header row if the 'header' option is
 * included.</p>
 *
 * <p>Possible options include:
 * <ul>
 * <li>csv - parse the table as comma-separated values</li>
 * <li>tsv - parse the table as tab-separated values</li>
 * <li>header - this table has a header (title) row</li>
 * </ul>
 * </p>
 *
 * <p>When passing in multiple options, pass them in as separate parameters,
 * seperated by commas. For example:
 * <br><br>
 * <code>
 *   loadTable("my_csv_file.csv", "csv", "header")
 * </code>
 * </p>
 *
 * <p> All files loaded and saved use UTF-8 encoding.</p>
 *
 * <p>This method is asynchronous, meaning it may not finish before the next
 * line in your sketch is executed. Calling loadTable() inside preload()
 * guarantees to complete the operation before setup() and draw() are called.
 * <p>Outside of preload(), you may supply a callback function to handle the
 * object:</p>
 * </p>
 *
 * @method loadTable
 * @param  {String}         filename   name of the file or URL to load
 * @param  {String|Strings} [options]  "header" "csv" "tsv"
 * @param  {Function}       [callback] function to be executed after
 *                                     loadTable() completes, Table object is
 *                                     passed in as first argument
 * @return {Object}                    Table object containing data
 *
 * @example
 * <div class="norender">
 * <code>
 * // Given the following CSV file called "mammals.csv"
 * // located in the project's "assets" folder:
 * //
 * // id,species,name
 * // 0,Capra hircus,Goat
 * // 1,Panthera pardus,Leopard
 * // 2,Equus zebra,Zebra
 *
 * var table;
 *
 * function preload() {
 *   //my table is comma separated value "csv"
 *   //and has a header specifying the columns labels
 *   table = loadTable("assets/mammals.csv", "csv", "header");
 *   //the file can be remote
 *   //table = loadTable("http://p5js.org/reference/assets/mammals.csv",
 *   //                  "csv", "header");
 * }
 *
 * function setup() {
 *   //count the columns
 *   print(table.getRowCount() + " total rows in table");
 *   print(table.getColumnCount() + " total columns in table");
 *
 *   print(table.getColumn("name"));
 *   //["Goat", "Leopard", "Zebra"]
 *
 *   //cycle through the table
 *   for (var r = 0; r < table.getRowCount(); r++)
 *     for (var c = 0; c < table.getColumnCount(); c++) {
 *       print(table.getString(r, c));
 *     }
 * }
 * </code>
 * </div>
 */
p5.prototype.loadTable = function (path) {
  var callback = null;
  var options = [];
  var header = false;
  var sep = ',';
  var separatorSet = false;
  var decrementPreload = p5._getDecrementPreload.apply(this, arguments);

  for (var i = 1; i < arguments.length; i++) {
    if ((typeof (arguments[i]) === 'function') &&
      (arguments[i] !== decrementPreload)) {
      callback = arguments[i];
    } else if (typeof (arguments[i]) === 'string') {
      options.push(arguments[i]);
      if (arguments[i] === 'header') {
        header = true;
      }
      if (arguments[i] === 'csv') {
        if (separatorSet) {
          throw new Error('Cannot set multiple separator types.');
        } else {
          sep = ',';
          separatorSet = true;
        }
      } else if (arguments[i] === 'tsv') {
        if (separatorSet) {
          throw new Error('Cannot set multiple separator types.');
        } else {
          sep = '\t';
          separatorSet = true;
        }
      }
    }
  }

  var t = new p5.Table();
  reqwest({
      url: path,
      crossOrigin: true,
      type: 'csv'
    })
    .then(function (resp) {
      resp = resp.responseText;

      var state = {};

      // define constants
      var PRE_TOKEN = 0,
        MID_TOKEN = 1,
        POST_TOKEN = 2,
        POST_RECORD = 4;

      var QUOTE = '\"',
        CR = '\r',
        LF = '\n';

      var records = [];
      var offset = 0;
      var currentRecord = null;
      var currentChar;

      var recordBegin = function () {
        state.escaped = false;
        currentRecord = [];
        tokenBegin();
      };

      var recordEnd = function () {
        state.currentState = POST_RECORD;
        records.push(currentRecord);
        currentRecord = null;
      };

      var tokenBegin = function () {
        state.currentState = PRE_TOKEN;
        state.token = '';
      };

      var tokenEnd = function () {
        currentRecord.push(state.token);
        tokenBegin();
      };

      while (true) {
        currentChar = resp[offset++];

        // EOF
        if (currentChar == null) {
          if (state.escaped) {
            throw new Error('Unclosed quote in file.');
          }
          if (currentRecord) {
            tokenEnd();
            recordEnd();
            break;
          }
        }
        if (currentRecord === null) {
          recordBegin();
        }

        // Handle opening quote
        if (state.currentState === PRE_TOKEN) {
          if (currentChar === QUOTE) {
            state.escaped = true;
            state.currentState = MID_TOKEN;
            continue;
          }
          state.currentState = MID_TOKEN;
        }

        // mid-token and escaped, look for sequences and end quote
        if (state.currentState === MID_TOKEN && state.escaped) {
          if (currentChar === QUOTE) {
            if (resp[offset] === QUOTE) {
              state.token += QUOTE;
              offset++;
            } else {
              state.escaped = false;
              state.currentState = POST_TOKEN;
            }
          } else {
            state.token += currentChar;
          }
          continue;
        }

        // fall-through: mid-token or post-token, not escaped
        if (currentChar === CR) {
          if (resp[offset] === LF) {
            offset++;
          }
          tokenEnd();
          recordEnd();
        } else if (currentChar === LF) {
          tokenEnd();
          recordEnd();
        } else if (currentChar === sep) {
          tokenEnd();
        } else if (state.currentState === MID_TOKEN) {
          state.token += currentChar;
        }
      }

      // set up column names
      if (header) {
        t.columns = records.shift();
      } else {
        for (i = 0; i < records[0].length; i++) {
          t.columns[i] = 'null';
        }
      }
      var row;
      for (i = 0; i < records.length; i++) {
        //Handles row of 'undefined' at end of some CSVs
        if (i === records.length - 1 && records[i].length === 1) {
          if (records[i][0] === 'undefined') {
            break;
          }
        }
        row = new p5.TableRow();
        row.arr = records[i];
        row.obj = makeObject(records[i], t.columns);
        t.addRow(row);
      }
      if (callback !== null) {
        callback(t);
      }
      if (decrementPreload && (callback !== decrementPreload)) {
        decrementPreload();
      }
    })
    .fail(function (err, msg) {
      p5._friendlyFileLoadError(2, path);
      // don't get error callback mixed up with decrementPreload
      if ((typeof callback !== 'undefined') &&
        (callback !== decrementPreload)) {
        callback(false);
      }
    });

  return t;
};

// helper function to turn a row into a JSON object
function makeObject(row, headers) {
  var ret = {};
  headers = headers || [];
  if (typeof (headers) === 'undefined') {
    for (var j = 0; j < row.length; j++) {
      headers[j.toString()] = j;
    }
  }
  for (var i = 0; i < headers.length; i++) {
    var key = headers[i];
    var val = row[i];
    ret[key] = val;
  }
  return ret;
}

/*global parseXML */
p5.prototype.parseXML = function (two) {
  var one = new p5.XML();
  var i;
  if (two.children.length) {
    for ( i = 0; i < two.children.length; i++ ) {
      var node = parseXML(two.children[i]);
      one.addChild(node);
    }
    one.setName(two.nodeName);
    one._setCont(two.textContent);
    one._setAttributes(two);
    for (var j = 0; j < one.children.length; j++) {
      one.children[j].parent = one;
    }
    return one;
  }
  else {
    one.setName(two.nodeName);
    one._setCont(two.textContent);
    one._setAttributes(two);
    return one;
  }
};

/**
 * Reads the contents of a file and creates an XML object with its values.
 * If the name of the file is used as the parameter, as in the above example,
 * the file must be located in the sketch directory/folder.
 *
 * Alternatively, the file maybe be loaded from anywhere on the local
 * computer using an absolute path (something that starts with / on Unix and
 * Linux, or a drive letter on Windows), or the filename parameter can be a
 * URL for a file found on a network.
 *
 * This method is asynchronous, meaning it may not finish before the next
 * line in your sketch is executed. Calling loadXML() inside preload()
 * guarantees to complete the operation before setup() and draw() are called.
 *
 * <p>Outside of preload(), you may supply a callback function to handle the
 * object:</p>
 *
 * @method loadXML
 * @param  {String}   filename   name of the file or URL to load
 * @param  {Function} [callback] function to be executed after loadXML()
 *                               completes, XML object is passed in as
 *                               first argument
 * @param  {Function} [errorCallback] function to be executed if
 *                               there is an error, response is passed
 *                               in as first argument
 * @return {Object}              XML object containing data
 */
p5.prototype.loadXML = function (path, callback, errorCallback) {
  var ret = {};
  var decrementPreload = p5._getDecrementPreload.apply(this, arguments);
  reqwest({
      url: path,
      type: 'xml',
      crossOrigin: true,
      error: function (resp) {
        // pass to error callback if defined
        if (errorCallback) {
          errorCallback(resp);
        } else { // otherwise log error msg
          console.log(resp.statusText);
        }
        //p5._friendlyFileLoadError(1,path);
      }
    })
    .then(function (resp) {
      var xml = parseXML(resp.documentElement);
      for(var key in xml) {
        ret[key] = xml[key];
      }
      if (typeof callback !== 'undefined') {
        callback(ret);
      }
      if (decrementPreload && (callback !== decrementPreload)) {
        decrementPreload();
      }
    });
  return ret;
};

// name clash with window.open
// p5.prototype.open = function() {
//   // TODO

// };

p5.prototype.selectFolder = function () {
  // TODO
  throw 'not yet implemented';

};

p5.prototype.selectInput = function () {
  // TODO
  throw 'not yet implemented';

};

/**
 * Method for executing an HTTP GET request. If data type is not specified,
 * p5 will try to guess based on the URL, defaulting to text.
 *
 * @method httpGet
 * @param  {String}        path       name of the file or url to load
 * @param  {Object}        [data]     param data passed sent with request
 * @param  {String}        [datatype] "json", "jsonp", "xml", or "text"
 * @param  {Function}      [callback] function to be executed after
 *                                    httpGet() completes, data is passed in
 *                                    as first argument
 * @param  {Function}      [errorCallback] function to be executed if
 *                                    there is an error, response is passed
 *                                    in as first argument
 */
p5.prototype.httpGet = function () {
  var args = new Array(arguments.length);
  for (var i = 0; i < args.length; ++i) {
    args[i] = arguments[i];
  }
  args.push('GET');
  p5.prototype.httpDo.apply(this, args);
};

/**
 * Method for executing an HTTP POST request. If data type is not specified,
 * p5 will try to guess based on the URL, defaulting to text.
 *
 * @method httpPost
 * @param  {String}        path       name of the file or url to load
 * @param  {Object}        [data]     param data passed sent with request
 * @param  {String}        [datatype] "json", "jsonp", "xml", or "text"
 * @param  {Function}      [callback] function to be executed after
 *                                    httpGet() completes, data is passed in
 *                                    as first argument
 * @param  {Function}      [errorCallback] function to be executed if
 *                                    there is an error, response is passed
 *                                    in as first argument
 */
p5.prototype.httpPost = function () {
  var args = new Array(arguments.length);
  for (var i = 0; i < args.length; ++i) {
    args[i] = arguments[i];
  }
  args.push('POST');
  p5.prototype.httpDo.apply(this, args);
};

/**
 * Method for executing an HTTP request. If data type is not specified,
 * p5 will try to guess based on the URL, defaulting to text.<br><br>
 * You may also pass a single object specifying all parameters for the
 * request following the examples inside the reqwest() calls here:
 * <a href='https://github.com/ded/reqwest#api'>
 * https://github.com/ded/reqwest#api</a>
 *
 * @method httpDo
 * @param  {String}        path       name of the file or url to load
 * @param  {String}        [method]   either "GET", "POST", or "PUT",
 *                                    defaults to "GET"
 * @param  {Object}        [data]     param data passed sent with request
 * @param  {String}        [datatype] "json", "jsonp", "xml", or "text"
 * @param  {Function}      [callback] function to be executed after
 *                                    httpGet() completes, data is passed in
 *                                    as first argument
 * @param  {Function}      [errorCallback] function to be executed if
 *                                    there is an error, response is passed
 *                                    in as first argument
 */
p5.prototype.httpDo = function () {
  if (typeof arguments[0] === 'object') {
    reqwest(arguments[0]);
  } else {
    var method = 'GET';
    var path = arguments[0];
    var data = {};
    var type = '';
    var callback;
    var errorCallback;

    for (var i = 1; i < arguments.length; i++) {
      var a = arguments[i];
      if (typeof a === 'string') {
        if (a === 'GET' || a === 'POST' || a === 'PUT') {
          method = a;
        } else {
          type = a;
        }
      } else if (typeof a === 'object') {
        data = a;
      } else if (typeof a === 'function') {
        if (!callback) {
          callback = a;
        } else {
          errorCallback = a;
        }
      }
    }

    // do some sort of smart type checking
    if (type === '') {
      if (path.indexOf('json') !== -1) {
        type = 'json';
      } else if (path.indexOf('xml') !== -1) {
        type = 'xml';
      } else {
        type = 'text';
      }
    }

    reqwest({
      url: path,
      method: method,
      data: data,
      type: type,
      crossOrigin: true,
      success: function (resp) {
        if (typeof callback !== 'undefined') {
          if (type === 'text') {
            callback(resp.response);
          } else {
            callback(resp);
          }
        }
      },
      error: function (resp) {
        if (errorCallback) {
          errorCallback(resp);
        } else {
          console.log(resp.statusText);
        }
      }
    });
  }
};

/**
 * @module IO
 * @submodule Output
 * @for p5
 */

window.URL = window.URL || window.webkitURL;

// private array of p5.PrintWriter objects
p5.prototype._pWriters = [];

p5.prototype.beginRaw = function () {
  // TODO
  throw 'not yet implemented';

};

p5.prototype.beginRecord = function () {
  // TODO
  throw 'not yet implemented';

};

p5.prototype.createOutput = function () {
  // TODO

  throw 'not yet implemented';
};

p5.prototype.createWriter = function (name, extension) {
  var newPW;
  // check that it doesn't already exist
  for (var i in p5.prototype._pWriters) {
    if (p5.prototype._pWriters[i].name === name) {
      // if a p5.PrintWriter w/ this name already exists...
      // return p5.prototype._pWriters[i]; // return it w/ contents intact.
      // or, could return a new, empty one with a unique name:
      newPW = new p5.PrintWriter(name + window.millis(), extension);
      p5.prototype._pWriters.push(newPW);
      return newPW;
    }
  }
  newPW = new p5.PrintWriter(name, extension);
  p5.prototype._pWriters.push(newPW);
  return newPW;
};

p5.prototype.endRaw = function () {
  // TODO

  throw 'not yet implemented';
};

p5.prototype.endRecord = function () {
  // TODO
  throw 'not yet implemented';

};

p5.PrintWriter = function (filename, extension) {
  var self = this;
  this.name = filename;
  this.content = '';
  this.print = function (data) {
    this.content += data;
  };
  this.println = function (data) {
    this.content += data + '\n';
  };
  this.flush = function () {
    this.content = '';
  };
  this.close = function () {
    // convert String to Array for the writeFile Blob
    var arr = [];
    arr.push(this.content);
    p5.prototype.writeFile(arr, filename, extension);
    // remove from _pWriters array and delete self
    for (var i in p5.prototype._pWriters) {
      if (p5.prototype._pWriters[i].name === this.name) {
        // remove from _pWriters array
        p5.prototype._pWriters.splice(i, 1);
      }
    }
    self.flush();
    self = {};
  };
};

p5.prototype.saveBytes = function () {
  // TODO
  throw 'not yet implemented';

};

// object, filename, options --> saveJSON, saveStrings, saveTable
// filename, [extension] [canvas] --> saveImage

/**
 *  <p>Save an image, text, json, csv, wav, or html. Prompts download to
 *  the client's computer. <b>Note that it is not recommended to call save()
 *  within draw if it's looping, as the save() function will open a new save
 *  dialog every frame.</b></p>
 *  <p>The default behavior is to save the canvas as an image. You can
 *  optionally specify a filename.
 *  For example:</p>
 *  <pre class='language-javascript'><code>
 *  save();
 *  save('myCanvas.jpg'); // save a specific canvas with a filename
 *  </code></pre>
 *
 *  <p>Alternately, the first parameter can be a pointer to a canvas
 *  p5.Element, an Array of Strings,
 *  an Array of JSON, a JSON object, a p5.Table, a p5.Image, or a
 *  p5.SoundFile (requires p5.sound). The second parameter is a filename
 *  (including extension). The third parameter is for options specific
 *  to this type of object. This method will save a file that fits the
 *  given paramaters. For example:</p>
 *
 *  <pre class='language-javascript'><code>
 *
 *  save('myCanvas.jpg');           // Saves canvas as an image
 *
 *  var cnv = createCanvas(100, 100);
 *  save(cnv, 'myCanvas.jpg');      // Saves canvas as an image
 *
 *  var gb = createGraphics(100, 100);
 *  save(gb, 'myGraphics.jpg');      // Saves p5.Renderer object as an image
 *
 *  save(myTable, 'myTable.html');  // Saves table as html file
 *  save(myTable, 'myTable.csv',);  // Comma Separated Values
 *  save(myTable, 'myTable.tsv');   // Tab Separated Values
 *
 *  save(myJSON, 'my.json');        // Saves pretty JSON
 *  save(myJSON, 'my.json', true);  // Optimizes JSON filesize
 *
 *  save(img, 'my.png');            // Saves pImage as a png image
 *
 *  save(arrayOfStrings, 'my.txt'); // Saves strings to a text file with line
 *                                  // breaks after each item in the array
 *  </code></pre>
 *
 *  @method save
 *  @param  {[Object|String]} objectOrFilename  If filename is provided, will
 *                                             save canvas as an image with
 *                                             either png or jpg extension
 *                                             depending on the filename.
 *                                             If object is provided, will
 *                                             save depending on the object
 *                                             and filename (see examples
 *                                             above).
 *  @param  {[String]} filename If an object is provided as the first
 *                               parameter, then the second parameter
 *                               indicates the filename,
 *                               and should include an appropriate
 *                               file extension (see examples above).
 *  @param  {[Boolean/String]} options  Additional options depend on
 *                            filetype. For example, when saving JSON,
 *                            <code>true</code> indicates that the
 *                            output will be optimized for filesize,
 *                            rather than readability.
 */
p5.prototype.save = function (object, _filename, _options) {
  // parse the arguments and figure out which things we are saving
  var args = arguments;
  // =================================================
  // OPTION 1: saveCanvas...

  // if no arguments are provided, save canvas
  var cnv = this._curElement.elt;
  if (args.length === 0) {
    p5.prototype.saveCanvas(cnv);
    return;
  }
  // otherwise, parse the arguments

  // if first param is a p5Graphics, then saveCanvas
  else if (args[0] instanceof p5.Renderer ||
    args[0] instanceof p5.Graphics) {
    p5.prototype.saveCanvas(args[0].elt, args[1], args[2]);
    return;
  }

  // if 1st param is String and only one arg, assume it is canvas filename
  else if (args.length === 1 && typeof (args[0]) === 'string') {
    p5.prototype.saveCanvas(cnv, args[0]);
  }

  // =================================================
  // OPTION 2: extension clarifies saveStrings vs. saveJSON
  else {
    var extension = _checkFileExtension(args[1], args[2])[1];
    switch (extension) {
    case 'json':
      p5.prototype.saveJSON(args[0], args[1], args[2]);
      return;
    case 'txt':
      p5.prototype.saveStrings(args[0], args[1], args[2]);
      return;
      // =================================================
      // OPTION 3: decide based on object...
    default:
      if (args[0] instanceof Array) {
        p5.prototype.saveStrings(args[0], args[1], args[2]);
      } else if (args[0] instanceof p5.Table) {
        p5.prototype.saveTable(args[0], args[1], args[2], args[3]);
      } else if (args[0] instanceof p5.Image) {
        p5.prototype.saveCanvas(args[0].canvas, args[1]);
      } else if (args[0] instanceof p5.SoundFile) {
        p5.prototype.saveSound(args[0], args[1], args[2], args[3]);
      }
    }
  }
};

/**
 *  Writes the contents of an Array or a JSON object to a .json file.
 *  The file saving process and location of the saved file will
 *  vary between web browsers.
 *
 *  @method saveJSON
 *  @param  {Array|Object} json
 *  @param  {String} filename
 *  @param  {Boolean} [optimize]   If true, removes line breaks
 *                                 and spaces from the output
 *                                 file to optimize filesize
 *                                 (but not readability).
 *  @example
 *  <div><code>
 *  var json;
 *
 *  function setup() {
 *
 *    json = {}; // new JSON Object
 *
 *    json.id = 0;
 *    json.species = 'Panthera leo';
 *    json.name = 'Lion';
 *
 *  // To save, un-comment the line below, then click 'run'
 *  // saveJSON(json, 'lion.json');
 *  }
 *
 *  // Saves the following to a file called "lion.json":
 *  // {
 *  //   "id": 0,
 *  //   "species": "Panthera leo",
 *  //   "name": "Lion"
 *  // }
 *  </div></code>
 */
p5.prototype.saveJSON = function (json, filename, opt) {
  var stringify;
  if (opt) {
    stringify = JSON.stringify(json);
  } else {
    stringify = JSON.stringify(json, undefined, 2);
  }
  console.log(stringify);
  this.saveStrings(stringify.split('\n'), filename, 'json');
};

p5.prototype.saveJSONObject = p5.prototype.saveJSON;
p5.prototype.saveJSONArray = p5.prototype.saveJSON;

p5.prototype.saveStream = function () {
  // TODO
  throw 'not yet implemented';

};

/**
 *  Writes an array of Strings to a text file, one line per String.
 *  The file saving process and location of the saved file will
 *  vary between web browsers.
 *
 *  @method saveStrings
 *  @param  {Array} list      string array to be written
 *  @param  {String} filename filename for output
 *  @example
 *  <div><code>
 *  var words = 'apple bear cat dog';
 *
 *  // .split() outputs an Array
 *  var list = split(words, ' ');
 *
 *  // To save the file, un-comment next line and click 'run'
 *  // saveStrings(list, 'nouns.txt');
 *
 *  // Saves the following to a file called 'nouns.txt':
 *  //
 *  // apple
 *  // bear
 *  // cat
 *  // dog
 *  </code></div>
 */
p5.prototype.saveStrings = function (list, filename, extension) {
  var ext = extension || 'txt';
  var pWriter = this.createWriter(filename, ext);
  for (var i = 0; i < list.length; i++) {
    if (i < list.length - 1) {
      pWriter.println(list[i]);
    } else {
      pWriter.print(list[i]);
    }
  }
  pWriter.close();
  pWriter.flush();
};

p5.prototype.saveXML = function () {
  // TODO
  throw 'not yet implemented';

};

p5.prototype.selectOutput = function () {
  // TODO
  throw 'not yet implemented';

};

// =======
// HELPERS
// =======

function escapeHelper(content) {
  return content
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
    .replace(/'/g, '&#039;');
}

/**
 *  Writes the contents of a Table object to a file. Defaults to a
 *  text file with comma-separated-values ('csv') but can also
 *  use tab separation ('tsv'), or generate an HTML table ('html').
 *  The file saving process and location of the saved file will
 *  vary between web browsers.
 *
 *  @method saveTable
 *  @param  {p5.Table} Table  the Table object to save to a file
 *  @param  {String} filename the filename to which the Table should be saved
 *  @param  {String} [options]  can be one of "tsv", "csv", or "html"
 *  @example
 *  <div><code>
 *  var table;
 *
 *  function setup() {
 *    table = new p5.Table();
 *
 *    table.addColumn('id');
 *    table.addColumn('species');
 *    table.addColumn('name');
 *
 *    var newRow = table.addRow();
 *    newRow.setNum('id', table.getRowCount() - 1);
 *    newRow.setString('species', 'Panthera leo');
 *    newRow.setString('name', 'Lion');
 *
 *    // To save, un-comment next line then click 'run'
 *    // saveTable(table, 'new.csv');
 *    }
 *
 *    // Saves the following to a file called 'new.csv':
 *    // id,species,name
 *    // 0,Panthera leo,Lion
 *  </code></div>
 */
p5.prototype.saveTable = function (table, filename, options) {
  var pWriter = this.createWriter(filename, options);

  var header = table.columns;

  var sep = ','; // default to CSV
  if (options === 'tsv') {
    sep = '\t';
  }
  if (options !== 'html') {
    // make header if it has values
    if (header[0] !== '0') {
      for (var h = 0; h < header.length; h++) {
        if (h < header.length - 1) {
          pWriter.print(header[h] + sep);
        } else {
          pWriter.println(header[h]);
        }
      }
    }

    // make rows
    for (var i = 0; i < table.rows.length; i++) {
      var j;
      for (j = 0; j < table.rows[i].arr.length; j++) {
        if (j < table.rows[i].arr.length - 1) {
          pWriter.print(table.rows[i].arr[j] + sep);
        } else if (i < table.rows.length - 1) {
          pWriter.println(table.rows[i].arr[j]);
        } else {
          pWriter.print(table.rows[i].arr[j]); // no line break
        }
      }
    }
  }

  // otherwise, make HTML
  else {
    pWriter.println('<html>');
    pWriter.println('<head>');
    var str = '  <meta http-equiv=\"content-type\" content';
    str += '=\"text/html;charset=utf-8\" />';
    pWriter.println(str);
    pWriter.println('</head>');

    pWriter.println('<body>');
    pWriter.println('  <table>');

    // make header if it has values
    if (header[0] !== '0') {
      pWriter.println('    <tr>');
      for (var k = 0; k < header.length; k++) {
        var e = escapeHelper(header[k]);
        pWriter.println('      <td>' + e);
        pWriter.println('      </td>');
      }
      pWriter.println('    </tr>');
    }

    // make rows
    for (var row = 0; row < table.rows.length; row++) {
      pWriter.println('    <tr>');
      for (var col = 0; col < table.columns.length; col++) {
        var entry = table.rows[row].getString(col);
        var htmlEntry = escapeHelper(entry);
        pWriter.println('      <td>' + htmlEntry);
        pWriter.println('      </td>');
      }
      pWriter.println('    </tr>');
    }
    pWriter.println('  </table>');
    pWriter.println('</body>');
    pWriter.print('</html>');
  }
  // close and flush the pWriter
  pWriter.close();
  pWriter.flush();
}; // end saveTable()

/**
 *  Generate a blob of file data as a url to prepare for download.
 *  Accepts an array of data, a filename, and an extension (optional).
 *  This is a private function because it does not do any formatting,
 *  but it is used by saveStrings, saveJSON, saveTable etc.
 *
 *  @param  {Array} dataToDownload
 *  @param  {String} filename
 *  @param  {[String]} extension
 *  @private
 */
p5.prototype.writeFile = function (dataToDownload, filename, extension) {
  var type = 'application\/octet-stream';
  if (p5.prototype._isSafari()) {
    type = 'text\/plain';
  }
  var blob = new Blob(dataToDownload, {
    'type': type
  });
  var href = window.URL.createObjectURL(blob);
  p5.prototype.downloadFile(href, filename, extension);
};

/**
 *  Forces download. Accepts a url to filedata/blob, a filename,
 *  and an extension (optional).
 *  This is a private function because it does not do any formatting,
 *  but it is used by saveStrings, saveJSON, saveTable etc.
 *
 *  @param  {String} href      i.e. an href generated by createObjectURL
 *  @param  {[String]} filename
 *  @param  {[String]} extension
 */
p5.prototype.downloadFile = function (href, fName, extension) {
  var fx = _checkFileExtension(fName, extension);
  var filename = fx[0];
  var ext = fx[1];

  var a = document.createElement('a');
  a.href = href;
  a.download = filename;

  // Firefox requires the link to be added to the DOM before click()
  a.onclick = destroyClickedElement;
  a.style.display = 'none';
  document.body.appendChild(a);

  // Safari will open this file in the same page as a confusing Blob.
  if (p5.prototype._isSafari()) {
    var aText = 'Hello, Safari user! To download this file...\n';
    aText += '1. Go to File --> Save As.\n';
    aText += '2. Choose "Page Source" as the Format.\n';
    aText += '3. Name it with this extension: .\"' + ext + '\"';
    alert(aText);
  }
  a.click();
  href = null;
};

/**
 *  Returns a file extension, or another string
 *  if the provided parameter has no extension.
 *
 *  @param   {String} filename
 *  @return  {Array} [fileName, fileExtension]
 *
 *  @private
 */
function _checkFileExtension(filename, extension) {
  if (!extension || extension === true || extension === 'true') {
    extension = '';
  }
  if (!filename) {
    filename = 'untitled';
  }
  var ext = '';
  // make sure the file will have a name, see if filename needs extension
  if (filename && filename.indexOf('.') > -1) {
    ext = filename.split('.').pop();
  }
  // append extension if it doesn't exist
  if (extension) {
    if (ext !== extension) {
      ext = extension;
      filename = filename + '.' + ext;
    }
  }
  return [filename, ext];
}
p5.prototype._checkFileExtension = _checkFileExtension;

/**
 *  Returns true if the browser is Safari, false if not.
 *  Safari makes trouble for downloading files.
 *
 *  @return  {Boolean} [description]
 *  @private
 */
p5.prototype._isSafari = function () {
  var x = Object.prototype.toString.call(window.HTMLElement);
  return x.indexOf('Constructor') > 0;
};

/**
 *  Helper function, a callback for download that deletes
 *  an invisible anchor element from the DOM once the file
 *  has been automatically downloaded.
 *
 *  @private
 */
function destroyClickedElement(event) {
  document.body.removeChild(event.target);
}

module.exports = p5;

},{"../core/core":37,"../core/error_helpers":40,"opentype.js":8,"reqwest":27}],60:[function(_dereq_,module,exports){
/**
 * @module IO
 * @submodule Table
 * @requires core
 */

'use strict';

var p5 = _dereq_('../core/core');


/**
 *  Table Options
 *  <p>Generic class for handling tabular data, typically from a
 *  CSV, TSV, or other sort of spreadsheet file.</p>
 *  <p>CSV files are
 *  <a href="http://en.wikipedia.org/wiki/Comma-separated_values">
 *  comma separated values</a>, often with the data in quotes. TSV
 *  files use tabs as separators, and usually don't bother with the
 *  quotes.</p>
 *  <p>File names should end with .csv if they're comma separated.</p>
 *  <p>A rough "spec" for CSV can be found
 *  <a href="http://tools.ietf.org/html/rfc4180">here</a>.</p>
 *  <p>To load files, use the loadTable method.</p>
 *  <p>To save tables to your computer, use the save method
 *   or the saveTable method.</p>
 *
 *  Possible options include:
 *  <ul>
 *  <li>csv - parse the table as comma-separated values
 *  <li>tsv - parse the table as tab-separated values
 *  <li>header - this table has a header (title) row
 *  </ul>
 */

/**
 *  Table objects store data with multiple rows and columns, much
 *  like in a traditional spreadsheet. Tables can be generated from
 *  scratch, dynamically, or using data from an existing file.
 *
 *  @class p5.Table
 *  @constructor
 *  @param  {Array}     [rows] An array of p5.TableRow objects
 *  @return {p5.Table}         p5.Table generated
 */
p5.Table = function (rows) {
  /**
   *  @property columns
   *  @type {Array}
   */
  this.columns = [];

  /**
   *  @property rows
   *  @type {Array}
   */
  this.rows = [];
};

/**
 *  Use addRow() to add a new row of data to a p5.Table object. By default,
 *  an empty row is created. Typically, you would store a reference to
 *  the new row in a TableRow object (see newRow in the example above),
 *  and then set individual values using set().
 *
 *  If a p5.TableRow object is included as a parameter, then that row is
 *  duplicated and added to the table.
 *
 *  @method  addRow
 *  @param   {p5.TableRow} [row] row to be added to the table
 *
 * @example
	* <div class="norender">
	* <code>
	* // Given the CSV file "mammals.csv"
	* // in the project's "assets" folder:
	* //
	* // id,species,name
	* // 0,Capra hircus,Goat
	* // 1,Panthera pardus,Leopard
	* // 2,Equus zebra,Zebra
	*
	* var table;
	*
	* function preload() {
	*   //my table is comma separated value "csv"
	*   //and has a header specifying the columns labels
	*   table = loadTable("assets/mammals.csv", "csv", "header");
	* }
	*
	* function setup() {
	*   //add a row
	*   var newRow = table.addRow();
	*   newRow.setString("id", table.getRowCount() - 1);
	*   newRow.setString("species", "Canis Lupus");
	*   newRow.setString("name", "Wolf");
	*
	*   //print the results
	*   for (var r = 0; r < table.getRowCount(); r++)
	*     for (var c = 0; c < table.getColumnCount(); c++)
	*       print(table.getString(r, c));
	* }
	* </code>
	* </div>
 */
p5.Table.prototype.addRow = function(row) {
  // make sure it is a valid TableRow
  var r = row || new p5.TableRow();

  if (typeof(r.arr) === 'undefined' || typeof(r.obj) === 'undefined') {
    //r = new p5.prototype.TableRow(r);
    throw 'invalid TableRow: ' + r;
  }
  r.table = this;
  this.rows.push(r);
  return r;
};

/**
 * Removes a row from the table object.
 *
 * @method  removeRow
 * @param   {Number} id ID number of the row to remove
 *
 * @example
	* <div class="norender">
	* <code>
	* // Given the CSV file "mammals.csv"
	* // in the project's "assets" folder:
	* //
	* // id,species,name
	* // 0,Capra hircus,Goat
	* // 1,Panthera pardus,Leopard
	* // 2,Equus zebra,Zebra
	*
	* var table;
	*
	* function preload() {
	*   //my table is comma separated value "csv"
	*   //and has a header specifying the columns labels
	*   table = loadTable("assets/mammals.csv", "csv", "header");
	* }
	*
	* function setup() {
	*   //remove the first row
	*   var r = table.removeRow(0);
	*
	*   //print the results
	*   for (var r = 0; r < table.getRowCount(); r++)
	*     for (var c = 0; c < table.getColumnCount(); c++)
	*       print(table.getString(r, c));
	* }
	* </code>
	* </div>
 */
p5.Table.prototype.removeRow = function(id) {
  this.rows[id].table = null; // remove reference to table
  var chunk = this.rows.splice(id+1, this.rows.length);
  this.rows.pop();
  this.rows = this.rows.concat(chunk);
};


/**
 * Returns a reference to the specified p5.TableRow. The reference
 * can then be used to get and set values of the selected row.
 *
 * @method  getRow
 * @param  {Number}   rowID ID number of the row to get
 * @return {TableRow} p5.TableRow object
 *
 * @example
	* <div class="norender">
	* <code>
	* // Given the CSV file "mammals.csv"
	* // in the project's "assets" folder:
	* //
	* // id,species,name
	* // 0,Capra hircus,Goat
	* // 1,Panthera pardus,Leopard
	* // 2,Equus zebra,Zebra
	*
	* var table;
	*
	* function preload() {
	*   //my table is comma separated value "csv"
	*   //and has a header specifying the columns labels
	*   table = loadTable("assets/mammals.csv", "csv", "header");
	* }
	*
	* function setup() {
	*   var row = table.getRow(1);
	*   //print it column by column
	*   //note: a row is an object, not an array
	*   for (var c = 0; c < table.getColumnCount(); c++)
	*     print(row.getString(c));
	* }
	* </code>
	* </div>
 */
p5.Table.prototype.getRow = function(r) {
  return this.rows[r];
};

/**
 *  Gets all rows from the table. Returns an array of p5.TableRows.
 *
 *  @method  getRows
 *  @return {Array}   Array of p5.TableRows
 *
 * @example
	* <div class="norender">
	* <code>
	* // Given the CSV file "mammals.csv"
	* // in the project's "assets" folder:
	* //
	* // id,species,name
	* // 0,Capra hircus,Goat
	* // 1,Panthera pardus,Leopard
	* // 2,Equus zebra,Zebra
	*
	* var table;
	*
	* function preload() {
	*   //my table is comma separated value "csv"
	*   //and has a header specifying the columns labels
	*   table = loadTable("assets/mammals.csv", "csv", "header");
	* }
	*
	* function setup() {
	*   var rows = table.getRows();
	*
	*   //warning: rows is an array of objects
	*   for (var r = 0; r < rows.length; r++)
	*     rows[r].set("name", "Unicorn");
	*
	*   //print the results
	*   for (var r = 0; r < table.getRowCount(); r++)
	*     for (var c = 0; c < table.getColumnCount(); c++)
	*       print(table.getString(r, c));
	* }
	* </code>
	* </div>
 */
p5.Table.prototype.getRows = function() {
  return this.rows;
};

/**
 *  Finds the first row in the Table that contains the value
 *  provided, and returns a reference to that row. Even if
 *  multiple rows are possible matches, only the first matching
 *  row is returned. The column to search may be specified by
 *  either its ID or title.
 *
 *  @method  findRow
 *  @param  {String} value  The value to match
 *  @param  {Number|String} column ID number or title of the
 *                                 column to search
 *  @return {TableRow}
 *
 * @example
	* <div class="norender">
	* <code>
	* // Given the CSV file "mammals.csv"
	* // in the project's "assets" folder:
	* //
	* // id,species,name
	* // 0,Capra hircus,Goat
	* // 1,Panthera pardus,Leopard
	* // 2,Equus zebra,Zebra
	*
	* var table;
	*
	* function preload() {
	*   //my table is comma separated value "csv"
	*   //and has a header specifying the columns labels
	*   table = loadTable("assets/mammals.csv", "csv", "header");
	* }
	*
	* function setup() {
	*   //find the animal named zebra
	*   var row = table.findRow("Zebra", "name");
	*   //find the corresponding species
	*   print(row.getString("species"));
	* }
	* </code>
	* </div>
 */
p5.Table.prototype.findRow = function(value, column) {
  // try the Object
  if (typeof(column) === 'string') {
    for (var i = 0; i < this.rows.length; i++){
      if (this.rows[i].obj[column] === value) {
        return this.rows[i];
      }
    }
  }
  // try the Array
  else {
    for (var j = 0; j < this.rows.length; j++){
      if (this.rows[j].arr[column] === value) {
        return this.rows[j];
      }
    }
  }
  // otherwise...
  return null;
};

/**
 *  Finds the rows in the Table that contain the value
 *  provided, and returns references to those rows. Returns an
 *  Array, so for must be used to iterate through all the rows,
 *  as shown in the example above. The column to search may be
 *  specified by either its ID or title.
 *
 *  @method  findRows
 *  @param  {String} value  The value to match
 *  @param  {Number|String} column ID number or title of the
 *                                 column to search
 *  @return {Array}        An Array of TableRow objects
 *
 * @example
	* <div class="norender">
	* <code>
	* // Given the CSV file "mammals.csv"
	* // in the project's "assets" folder:
	* //
	* // id,species,name
	* // 0,Capra hircus,Goat
	* // 1,Panthera pardus,Leopard
	* // 2,Equus zebra,Zebra
	*
	* var table;
	*
	* function preload() {
	*   //my table is comma separated value "csv"
	*   //and has a header specifying the columns labels
	*   table = loadTable("assets/mammals.csv", "csv", "header");
	* }
	*
	* function setup() {
	*   //add another goat
	*   var newRow = table.addRow();
	*   newRow.setString("id", table.getRowCount() - 1);
	*   newRow.setString("species", "Scape Goat");
	*   newRow.setString("name", "Goat");
	*
	*   //find the rows containing animals named Goat
	*   var rows = table.findRows("Goat", "name");
	*   print(rows.length + " Goats found");
	* }
	* </code>
	* </div>
 */
p5.Table.prototype.findRows = function(value, column) {
  var ret = [];
  if (typeof(column) === 'string') {
    for (var i = 0; i < this.rows.length; i++){
      if (this.rows[i].obj[column] === value) {
        ret.push( this.rows[i] );
      }
    }
  }
  // try the Array
  else {
    for (var j = 0; j < this.rows.length; j++){
      if (this.rows[j].arr[column] === value) {
        ret.push( this.rows[j] );
      }
    }
  }
  return ret;
};

/**
 *  Finds the first row in the Table that matches the regular
 *  expression provided, and returns a reference to that row.
 *  Even if multiple rows are possible matches, only the first
 *  matching row is returned. The column to search may be
 *  specified by either its ID or title.
 *
 *  @method  matchRow
 *  @param  {String} regexp The regular expression to match
 *  @param  {String|Number} column The column ID (number) or
 *                                   title (string)
 *  @return {TableRow}        TableRow object
 */
p5.Table.prototype.matchRow = function(regexp, column) {
  if (typeof(column) === 'number') {
    for (var j = 0; j < this.rows.length; j++) {
      if ( this.rows[j].arr[column].match(regexp) ) {
        return this.rows[j];
      }
    }
  }

  else {
    for (var i = 0; i < this.rows.length; i++) {
      if ( this.rows[i].obj[column].match(regexp) ) {
        return this.rows[i];
      }
    }
  }
  return null;
};

/**
 *  Finds the rows in the Table that match the regular expression provided,
 *  and returns references to those rows. Returns an array, so for must be
 *  used to iterate through all the rows, as shown in the example. The
 *  column to search may be specified by either its ID or title.
 *
 *  @method  matchRows
 *  @param  {String} regexp The regular expression to match
 *  @param  {String|Number} [column] The column ID (number) or
 *                                   title (string)
 *  @return {Array}        An Array of TableRow objects
 *  @example
 *  var table;
 *
 *  function setup() {
 *
 *    table = new p5.Table();
 *
 *    table.addColumn('name');
 *    table.addColumn('type');
 *
 *    var newRow = table.addRow();
 *    newRow.setString('name', 'Lion');
 *    newRow.setString('type', 'Mammal');
 *
 *    newRow = table.addRow();
 *    newRow.setString('name', 'Snake');
 *    newRow.setString('type', 'Reptile');
 *
 *    newRow = table.addRow();
 *    newRow.setString('name', 'Mosquito');
 *    newRow.setString('type', 'Insect');
 *
 *    newRow = table.addRow();
 *    newRow.setString('name', 'Lizard');
 *    newRow.setString('type', 'Reptile');
 *
 *    var rows = table.matchRows('R.*', 'type');
 *    for (var i = 0; i < rows.length; i++) {
 *      println(rows[i].getString('name') + ': ' + rows[i].getString('type'));
 *    }
 *  }
 *  // Sketch prints:
 *  // Snake: Reptile
 *  // Lizard: Reptile
 */
p5.Table.prototype.matchRows = function(regexp, column) {
  var ret = [];
  if (typeof(column) === 'number') {
    for (var j = 0; j < this.rows.length; j++) {
      if ( this.rows[j].arr[column].match(regexp) ) {
        ret.push( this.rows[j] );
      }
    }
  }

  else {
    for (var i = 0; i < this.rows.length; i++) {
      if ( this.rows[i].obj[column].match(regexp) ) {
        ret.push( this.rows[i] );
      }
    }
  }
  return ret;
};


/**
 *  Retrieves all values in the specified column, and returns them
 *  as an array. The column may be specified by either its ID or title.
 *
 *  @method  getColumn
 *  @param  {String|Number} column String or Number of the column to return
 *  @return {Array}       Array of column values
 *
 * @example
	* <div class="norender">
	* <code>
	* // Given the CSV file "mammals.csv"
	* // in the project's "assets" folder:
	* //
	* // id,species,name
	* // 0,Capra hircus,Goat
	* // 1,Panthera pardus,Leopard
	* // 2,Equus zebra,Zebra
	*
	* var table;
	*
	* function preload() {
	*   //my table is comma separated value "csv"
	*   //and has a header specifying the columns labels
	*   table = loadTable("assets/mammals.csv", "csv", "header");
	* }
	*
	* function setup() {
	*   //getColumn returns an array that can be printed directly
	*   print(table.getColumn("species"));
	*   //outputs ["Capra hircus", "Panthera pardus", "Equus zebra"]
	* }
	* </code>
	* </div>
 */
p5.Table.prototype.getColumn = function(value) {
  var ret = [];
  if (typeof(value) === 'string'){
    for (var i = 0; i < this.rows.length; i++){
      ret.push (this.rows[i].obj[value]);
    }
  } else {
    for (var j = 0; j < this.rows.length; j++){
      ret.push (this.rows[j].arr[value]);
    }
  }
  return ret;
};

/**
 *  Removes all rows from a Table. While all rows are removed,
 *  columns and column titles are maintained.
 *
 *  @method  clearRows
 *
 * @example
	* <div class="norender">
	* <code>
	* // Given the CSV file "mammals.csv"
	* // in the project's "assets" folder:
	* //
	* // id,species,name
	* // 0,Capra hircus,Goat
	* // 1,Panthera pardus,Leopard
	* // 2,Equus zebra,Zebra
	*
	* var table;
	*
	* function preload() {
	*   //my table is comma separated value "csv"
	*   //and has a header specifying the columns labels
	*   table = loadTable("assets/mammals.csv", "csv", "header");
	* }
	*
	* function setup() {
	*   table.clearRows();
	*   print(table.getRowCount() + " total rows in table");
	*   print(table.getColumnCount() + " total columns in table");
	* }
	* </code>
	* </div>
 */
p5.Table.prototype.clearRows = function() {
  delete this.rows;
  this.rows = [];
};

/**
 *  Use addColumn() to add a new column to a Table object.
 *  Typically, you will want to specify a title, so the column
 *  may be easily referenced later by name. (If no title is
 *  specified, the new column's title will be null.)
 *
 *  @method  addColumn
 *  @param {String} [title] title of the given column
 *
 * @example
	* <div class="norender">
	* <code>
	* // Given the CSV file "mammals.csv"
	* // in the project's "assets" folder:
	* //
	* // id,species,name
	* // 0,Capra hircus,Goat
	* // 1,Panthera pardus,Leopard
	* // 2,Equus zebra,Zebra
	*
	* var table;
	*
	* function preload() {
	*   //my table is comma separated value "csv"
	*   //and has a header specifying the columns labels
	*   table = loadTable("assets/mammals.csv", "csv", "header");
	* }
	*
	* function setup() {
	*   table.addColumn("carnivore");
	*   table.set(0, "carnivore", "no");
	*   table.set(1, "carnivore", "yes");
	*   table.set(2, "carnivore", "no");
	*
	*   //print the results
	*   for (var r = 0; r < table.getRowCount(); r++)
	*     for (var c = 0; c < table.getColumnCount(); c++)
	*       print(table.getString(r, c));
	* }
	* </code>
	* </div>
 */
p5.Table.prototype.addColumn = function(title) {
  var t = title || null;
  this.columns.push(t);
};

/**
 *  Returns the total number of columns in a Table.
 *
 *  @return {Number} Number of columns in this table
 */
p5.Table.prototype.getColumnCount = function() {
  return this.columns.length;
};

/**
 *  Returns the total number of rows in a Table.
 *
 *  @method  getRowCount
 *  @return {Number} Number of rows in this table

 */
p5.Table.prototype.getRowCount = function() {
  return this.rows.length;
};

/**
 *  <p>Removes any of the specified characters (or "tokens").</p>
 *
 *  <p>If no column is specified, then the values in all columns and
 *  rows are processed. A specific column may be referenced by
 *  either its ID or title.</p>
 *
 *  @method  removeTokens
 *  @param  {String} chars  String listing characters to be removed
 *  @param  {String|Number} [column] Column ID (number)
 *                                   or name (string)
 */
p5.Table.prototype.removeTokens = function(chars, column) {
  var escape= function(s) {
    return s.replace(/[-\/\\^$*+?.()|[\]{}]/g, '\\$&');
  };
  var charArray = [];
  for (var i = 0; i < chars.length; i++) {
    charArray.push( escape( chars.charAt(i) ) );
  }
  var regex = new RegExp(charArray.join('|'), 'g');

  if (typeof(column) === 'undefined'){
    for (var c = 0; c < this.columns.length; c++) {
      for (var d = 0; d < this.rows.length; d++) {
        var s = this.rows[d].arr[c];
        s = s.replace(regex, '');
        this.rows[d].arr[c] = s;
        this.rows[d].obj[this.columns[c]] = s;
      }
    }
  }
  else if (typeof(column) === 'string'){
    for (var j = 0; j < this.rows.length; j++) {
      var val = this.rows[j].obj[column];
      val = val.replace(regex, '');
      this.rows[j].obj[column] = val;
      var pos = this.columns.indexOf(column);
      this.rows[j].arr[pos] = val;
    }
  }
  else {
    for (var k = 0; k < this.rows.length; k++) {
      var str = this.rows[k].arr[column];
      str = str.replace(regex, '');
      this.rows[k].arr[column] = str;
      this.rows[k].obj[this.columns[column]] = str;
    }
  }
};

/**
 *  Trims leading and trailing whitespace, such as spaces and tabs,
 *  from String table values. If no column is specified, then the
 *  values in all columns and rows are trimmed. A specific column
 *  may be referenced by either its ID or title.
 *
 *  @method  trim
 *  @param  {String|Number} column Column ID (number)
 *                                   or name (string)
 */
p5.Table.prototype.trim = function(column) {
  var regex = new RegExp( (' '), 'g');

  if (typeof(column) === 'undefined'){
    for (var c = 0; c < this.columns.length; c++) {
      for (var d = 0; d < this.rows.length; d++) {
        var s = this.rows[d].arr[c];
        s = s.replace(regex, '');
        this.rows[d].arr[c] = s;
        this.rows[d].obj[this.columns[c]] = s;
      }
    }
  }
  else if (typeof(column) === 'string'){
    for (var j = 0; j < this.rows.length; j++) {
      var val = this.rows[j].obj[column];
      val = val.replace(regex, '');
      this.rows[j].obj[column] = val;
      var pos = this.columns.indexOf(column);
      this.rows[j].arr[pos] = val;
    }
  }
  else {
    for (var k = 0; k < this.rows.length; k++) {
      var str = this.rows[k].arr[column];
      str = str.replace(regex, '');
      this.rows[k].arr[column] = str;
      this.rows[k].obj[this.columns[column]] = str;
    }
  }
};

/**
 *  Use removeColumn() to remove an existing column from a Table
 *  object. The column to be removed may be identified by either
 *  its title (a String) or its index value (an int).
 *  removeColumn(0) would remove the first column, removeColumn(1)
 *  would remove the second column, and so on.
 *
 *  @method  removeColumn
 *  @param  {String|Number} column columnName (string) or ID (number)
 *
 * @example
	* <div class="norender">
	* <code>
	* // Given the CSV file "mammals.csv"
	* // in the project's "assets" folder:
	* //
	* // id,species,name
	* // 0,Capra hircus,Goat
	* // 1,Panthera pardus,Leopard
	* // 2,Equus zebra,Zebra
	*
	* var table;
	*
	* function preload() {
	*   //my table is comma separated value "csv"
	*   //and has a header specifying the columns labels
	*   table = loadTable("assets/mammals.csv", "csv", "header");
	* }
	*
	* function setup() {
	*   table.removeColumn("id");
	*   print(table.getColumnCount());
	* }
	* </code>
	* </div>
 */
p5.Table.prototype.removeColumn = function(c) {
  var cString;
  var cNumber;
  if (typeof(c) === 'string') {
    // find the position of c in the columns
    cString = c;
    cNumber = this.columns.indexOf(c);
    console.log('string');
  }
  else{
    cNumber = c;
    cString = this.columns[c];
  }

  var chunk = this.columns.splice(cNumber+1, this.columns.length);
  this.columns.pop();
  this.columns = this.columns.concat(chunk);

  for (var i = 0; i < this.rows.length; i++){
    var tempR = this.rows[i].arr;
    var chip = tempR.splice(cNumber+1, tempR.length);
    tempR.pop();
    this.rows[i].arr = tempR.concat(chip);
    delete this.rows[i].obj[cString];
  }

};


/**
 * Stores a value in the Table's specified row and column.
 * The row is specified by its ID, while the column may be specified
 * by either its ID or title.
 *
 * @method  set
 * @param {String|Number} column column ID (Number)
 *                               or title (String)
 * @param {String|Number} value  value to assign
 *
 * @example
	* <div class="norender">
	* <code>
	* // Given the CSV file "mammals.csv"
	* // in the project's "assets" folder:
	* //
	* // id,species,name
	* // 0,Capra hircus,Goat
	* // 1,Panthera pardus,Leopard
	* // 2,Equus zebra,Zebra
	*
	* var table;
	*
	* function preload() {
	*   //my table is comma separated value "csv"
	*   //and has a header specifying the columns labels
	*   table = loadTable("assets/mammals.csv", "csv", "header");
	* }
	*
	* function setup() {
	*   table.set(0, "species", "Canis Lupus");
	*   table.set(0, "name", "Wolf");
	*
	*   //print the results
	*   for (var r = 0; r < table.getRowCount(); r++)
	*     for (var c = 0; c < table.getColumnCount(); c++)
	*       print(table.getString(r, c));
	* }
	* </code>
	* </div>
 */
p5.Table.prototype.set = function(row, column, value) {
  this.rows[row].set(column, value);
};

/**
 * Stores a Float value in the Table's specified row and column.
 * The row is specified by its ID, while the column may be specified
 * by either its ID or title.
 *
 * @method setNum
 * @param {Number} row row ID
 * @param {String|Number} column column ID (Number)
 *                               or title (String)
 * @param {Number} value  value to assign
 *
 * @example
	* <div class="norender">
	* <code>
	* // Given the CSV file "mammals.csv"
	* // in the project's "assets" folder:
	* //
	* // id,species,name
	* // 0,Capra hircus,Goat
	* // 1,Panthera pardus,Leopard
	* // 2,Equus zebra,Zebra
	*
	* var table;
	*
	* function preload() {
	*   //my table is comma separated value "csv"
	*   //and has a header specifying the columns labels
	*   table = loadTable("assets/mammals.csv", "csv", "header");
	* }
	*
	* function setup() {
	*   table.setNum(1, "id", 1);
	*
	*   print(table.getColumn(0));
	*   //["0", 1, "2"]
	* }
	* </code>
	* </div>
 */
p5.Table.prototype.setNum = function(row, column, value){
  this.rows[row].setNum(column, value);
};


/**
 * Stores a String value in the Table's specified row and column.
 * The row is specified by its ID, while the column may be specified
 * by either its ID or title.
 *
 * @method  setString
 * @param {Number} row row ID
 * @param {String|Number} column column ID (Number)
 *                               or title (String)
 * @param {String} value  value to assign
 */
p5.Table.prototype.setString = function(row, column, value){
  this.rows[row].setString(column, value);
};

/**
 * Retrieves a value from the Table's specified row and column.
 * The row is specified by its ID, while the column may be specified by
 * either its ID or title.
 *
 * @method  get
 * @param {Number} row row ID
 * @param  {String|Number} column columnName (string) or
 *                                   ID (number)
 * @return {String|Number}
 *
 * @example
	* <div class="norender">
	* <code>
	* // Given the CSV file "mammals.csv"
	* // in the project's "assets" folder:
	* //
	* // id,species,name
	* // 0,Capra hircus,Goat
	* // 1,Panthera pardus,Leopard
	* // 2,Equus zebra,Zebra
	*
	* var table;
	*
	* function preload() {
	*   //my table is comma separated value "csv"
	*   //and has a header specifying the columns labels
	*   table = loadTable("assets/mammals.csv", "csv", "header");
	* }
	*
	* function setup() {
	*   print(table.get(0, 1));
	*   //Capra hircus
	*   print(table.get(0, "species"));
	*   //Capra hircus
	* }
	* </code>
	* </div>
 */
p5.Table.prototype.get = function(row, column) {
  return this.rows[row].get(column);
};

/**
 * Retrieves a Float value from the Table's specified row and column.
 * The row is specified by its ID, while the column may be specified by
 * either its ID or title.
 *
 * @method  getNum
 * @param {Number} row row ID
 * @param  {String|Number} column columnName (string) or
 *                                   ID (number)
 * @return {Number}
 *
 * @example
	* <div class="norender">
	* <code>
	* // Given the CSV file "mammals.csv"
	* // in the project's "assets" folder:
	* //
	* // id,species,name
	* // 0,Capra hircus,Goat
	* // 1,Panthera pardus,Leopard
	* // 2,Equus zebra,Zebra
	*
	* var table;
	*
	* function preload() {
	*   //my table is comma separated value "csv"
	*   //and has a header specifying the columns labels
	*   table = loadTable("assets/mammals.csv", "csv", "header");
	* }
	*
	* function setup() {
	*   print(table.getNum(1, 0) + 100);
	*   //id 1 + 100 = 101
	* }
	* </code>
	* </div>
 */
p5.Table.prototype.getNum = function(row, column) {
  return this.rows[row].getNum(column);
};

/**
 * Retrieves a String value from the Table's specified row and column.
 * The row is specified by its ID, while the column may be specified by
 * either its ID or title.
 *
 * @method  getString
 * @param {Number} row row ID
 * @param  {String|Number} column columnName (string) or
 *                                   ID (number)
 * @return {String}
 *
 * @example
	* <div class="norender">
	* <code>
	* // Given the CSV file "mammals.csv"
	* // in the project's "assets" folder:
	* //
	* // id,species,name
	* // 0,Capra hircus,Goat
	* // 1,Panthera pardus,Leopard
	* // 2,Equus zebra,Zebra
	*
	* var table;
	*
	* function preload() {
	*   //my table is comma separated value "csv"
	*   //and has a header specifying the columns labels
	*   table = loadTable("assets/mammals.csv", "csv", "header");
	* }
	*
	* function setup() {
	*   var tableArray = table.getArray();
	*
	*   //output each row as array
	*   for (var i = 0; i < tableArray.length; i++)
	*     print(tableArray[i]);
	* }
	* </code>
	* </div>
 */
p5.Table.prototype.getString = function(row, column) {
  return this.rows[row].getString(column);
};

/**
 * Retrieves all table data and returns as an object. If a column name is
 * passed in, each row object will be stored with that attribute as its
 * title.
 *
 * @method  getObject
 * @param {String} headerColumn Name of the column which should be used to
 *                              title each row object (optional)
 * @return {Object}
 *
 * @example
	* <div class="norender">
	* <code>
	* // Given the CSV file "mammals.csv"
	* // in the project's "assets" folder:
	* //
	* // id,species,name
	* // 0,Capra hircus,Goat
	* // 1,Panthera pardus,Leopard
	* // 2,Equus zebra,Zebra
	*
	* var table;
	*
	* function preload() {
	*   //my table is comma separated value "csv"
	*   //and has a header specifying the columns labels
	*   table = loadTable("assets/mammals.csv", "csv", "header");
	* }
	*
	* function setup() {
	*   var tableObject = table.getObject();
	*
	*   print(tableObject);
	*   //outputs an object
	* }
	* </code>
	* </div>

 */
p5.Table.prototype.getObject = function (headerColumn) {
  var tableObject = {};
  var obj, cPos, index;

  for(var i = 0; i < this.rows.length; i++) {
    obj = this.rows[i].obj;

    if (typeof(headerColumn) === 'string'){
      cPos = this.columns.indexOf(headerColumn); // index of columnID
      if (cPos >= 0) {
        index = obj[headerColumn];
        tableObject[index] = obj;
      } else {
        throw 'This table has no column named "' + headerColumn +'"';
      }
    } else {
      tableObject[i] = this.rows[i].obj;
    }
  }
  return tableObject;
};

/**
 * Retrieves all table data and returns it as a multidimensional array.
 *
 * @method  getArray
 * @return {Array}
 */
p5.Table.prototype.getArray = function () {
  var tableArray = [];
  for(var i = 0; i < this.rows.length; i++) {
    tableArray.push(this.rows[i].arr);
  }
  return tableArray;
};

module.exports = p5.Table;

},{"../core/core":37}],61:[function(_dereq_,module,exports){
/**
 * @module IO
 * @submodule Table
 * @requires core
 */

'use strict';

var p5 = _dereq_('../core/core');

/**
 *  A TableRow object represents a single row of data values,
 *  stored in columns, from a table.
 *
 *  A Table Row contains both an ordered array, and an unordered
 *  JSON object.
 *
 *  @class p5.TableRow
 *  @constructor
 *  @param {String} [str]       optional: populate the row with a
 *                              string of values, separated by the
 *                              separator
 *  @param {String} [separator] comma separated values (csv) by default
 */
p5.TableRow = function (str, separator) {
  var arr = [];
  var obj = {};
  if (str){
    separator = separator || ',';
    arr = str.split(separator);
  }
  for (var i = 0; i < arr.length; i++){
    var key = i;
    var val = arr[i];
    obj[key] = val;
  }
  this.arr = arr;
  this.obj = obj;
  this.table = null;
};

/**
 *  Stores a value in the TableRow's specified column.
 *  The column may be specified by either its ID or title.
 *
 *  @method  set
 *  @param {String|Number} column Column ID (Number)
 *                                or Title (String)
 *  @param {String|Number} value  The value to be stored
 */
p5.TableRow.prototype.set = function(column, value) {
  // if typeof column is string, use .obj
  if (typeof(column) === 'string'){
    var cPos = this.table.columns.indexOf(column); // index of columnID
    if (cPos >= 0) {
      this.obj[column] = value;
      this.arr[cPos] = value;
    }
    else {
      throw 'This table has no column named "' + column +'"';
    }
  }

  // if typeof column is number, use .arr
  else {
    if (column < this.table.columns.length) {
      this.arr[column] = value;
      var cTitle = this.table.columns[column];
      this.obj[cTitle] = value;
    }
    else {
      throw 'Column #' + column + ' is out of the range of this table';
    }
  }
};


/**
 *  Stores a Float value in the TableRow's specified column.
 *  The column may be specified by either its ID or title.
 *
 *  @method  setNum
 *  @param {String|Number} column Column ID (Number)
 *                                or Title (String)
 *  @param {Number} value  The value to be stored
 *                                as a Float
 */
p5.TableRow.prototype.setNum = function(column, value){
  var floatVal = parseFloat(value, 10);
  this.set(column, floatVal);
};


/**
 *  Stores a String value in the TableRow's specified column.
 *  The column may be specified by either its ID or title.
 *
 *  @method  setString
 *  @param {String|Number} column Column ID (Number)
 *                                or Title (String)
 *  @param {String} value  The value to be stored
 *                                as a String
 */
p5.TableRow.prototype.setString = function(column, value){
  var stringVal = value.toString();
  this.set(column, stringVal);
};

/**
 *  Retrieves a value from the TableRow's specified column.
 *  The column may be specified by either its ID or title.
 *
 *  @method  get
 *  @param  {String|Number} column columnName (string) or
 *                                   ID (number)
 *  @return {String|Number}
 */
p5.TableRow.prototype.get = function(column) {
  if (typeof(column) === 'string'){
    return this.obj[column];
  } else {
    return this.arr[column];
  }
};

/**
 *  Retrieves a Float value from the TableRow's specified
 *  column. The column may be specified by either its ID or
 *  title.
 *
 *  @method  getNum
 *  @param  {String|Number} column columnName (string) or
 *                                   ID (number)
 *  @return {Number}  Float Floating point number
 */
p5.TableRow.prototype.getNum = function(column) {
  var ret;
  if (typeof(column) === 'string'){
    ret = parseFloat(this.obj[column], 10);
  } else {
    ret = parseFloat(this.arr[column], 10);
  }

  if (ret.toString() === 'NaN') {
    throw 'Error: ' + this.obj[column]+ ' is NaN (Not a Number)';
  }
  return ret;
};

/**
 *  Retrieves an String value from the TableRow's specified
 *  column. The column may be specified by either its ID or
 *  title.
 *
 *  @method  getString
 *  @param  {String|Number} column columnName (string) or
 *                                   ID (number)
 *  @return {String}  String
 */
p5.TableRow.prototype.getString = function(column) {
  if (typeof(column) === 'string'){
    return this.obj[column].toString();
  } else {
    return this.arr[column].toString();
  }
};

module.exports = p5.TableRow;

},{"../core/core":37}],62:[function(_dereq_,module,exports){
/**
 * @module IO
 * @submodule XML
 * @requires core
 */

'use strict';

var p5 = _dereq_('../core/core');

/**
 * XML is a representation of an XML object, able to parse XML code. Use
 * loadXML() to load external XML files and create XML objects.
 *
 * @class p5.XML
 * @constructor
 * @return {p5.XML}    p5.XML object generated
 * @example
 * <div class='norender'><code>
 * // The following short XML file called "mammals.xml" is parsed
 * // in the code below.
 * //
 * // <?xml version="1.0"?>
 * // &lt;mammals&gt;
 * //   &lt;animal id="0" species="Capra hircus">Goat&lt;/animal&gt;
 * //   &lt;animal id="1" species="Panthera pardus">Leopard&lt;/animal&gt;
 * //   &lt;animal id="2" species="Equus zebra">Zebra&lt;/animal&gt;
 * // &lt;/mammals&gt;
 *
 * var xml;
 *
 * function preload() {
 *   xml = loadXML("assets/mammals.xml");
 * }
 *
 * function setup() {
 *   var children = xml.getChildren("animal");
 *
 *   for (var i = 0; i < children.length; i++) {
 *     var id = children[i].getNumber("id");
 *     var coloring = children[i].getString("species");
 *     var name = children[i].getContent();
 *     print(id + ", " + coloring + ", " + name);
 *   }
 * }
 *
 * // Sketch prints:
 * // 0, Capra hircus, Goat
 * // 1, Panthera pardus, Leopard
 * // 2, Equus zebra, Zebra
 * </code></div>
 */
p5.XML = function () {
  this.name = null; //done
  this.attributes = {}; //done
  this.children = [];
  this.parent = null;
  this.content = null; //done
};


/**
 * Gets a copy of the element's parent. Returns the parent as another
 * p5.XML object.
 *
 * @method getParent
 * @return {Object}   element parent
 * @example
 * <div class='norender'><code>
 * // The following short XML file called "mammals.xml" is parsed
 * // in the code below.
 * //
 * // <?xml version="1.0"?>
 * // &lt;mammals&gt;
 * //   &lt;animal id="0" species="Capra hircus">Goat&lt;/animal&gt;
 * //   &lt;animal id="1" species="Panthera pardus">Leopard&lt;/animal&gt;
 * //   &lt;animal id="2" species="Equus zebra">Zebra&lt;/animal&gt;
 * // &lt;/mammals&gt;
 *
 * var xml;
 *
 * function preload() {
 *   xml = loadXML("assets/mammals.xml");
 * }
 *
 * function setup() {
 *   var children = xml.getChildren("animal");
 *   var parent = children[1].getParent();
 *   print(parent.getName());
 * }
 *
 * // Sketch prints:
 * // mammals
 * </code></div>
 */
p5.XML.prototype.getParent = function() {
  return this.parent;
};

/**
 *  Gets the element's full name, which is returned as a String.
 *
 * @method getName
 * @return {String} the name of the node
 * @example&lt;animal
 * <div class='norender'><code>
 * // The following short XML file called "mammals.xml" is parsed
 * // in the code below.
 * //
 * // <?xml version="1.0"?>
 * // &lt;mammals&gt;
 * //   &lt;animal id="0" species="Capra hircus">Goat&lt;/animal&gt;
 * //   &lt;animal id="1" species="Panthera pardus">Leopard&lt;/animal&gt;
 * //   &lt;animal id="2" species="Equus zebra">Zebra&lt;/animal&gt;
 * // &lt;/mammals&gt;
 *
 * var xml;
 *
 * function preload() {
 *   xml = loadXML("assets/mammals.xml");
 * }
 *
 * function setup() {
 *   println(xml.getName());
 * }
 *
 * // Sketch prints:
 * // mammals
 * </code></div>
 */
p5.XML.prototype.getName = function() {
  return this.name;
};

/**
 * Sets the element's name, which is specified as a String.
 *
 * @method setName
 * @param {String} the new name of the node
 * @example&lt;animal
 * <div class='norender'><code>
 * // The following short XML file called "mammals.xml" is parsed
 * // in the code below.
 * //
 * // <?xml version="1.0"?>
 * // &lt;mammals&gt;
 * //   &lt;animal id="0" species="Capra hircus">Goat&lt;/animal&gt;
 * //   &lt;animal id="1" species="Panthera pardus">Leopard&lt;/animal&gt;
 * //   &lt;animal id="2" species="Equus zebra">Zebra&lt;/animal&gt;
 * // &lt;/mammals&gt;
 *
 * var xml;
 *
 * function preload() {
 *   xml = loadXML("assets/mammals.xml");
 * }
 *
 * function setup() {
 *   print(xml.getName());
 *   xml.setName("fish");
 *   print(xml.getName());
 * }
 *
 * // Sketch prints:
 * // mammals
 * // fish
 * </code></div>
 */
p5.XML.prototype.setName = function(name) {
  this.name = name;
};

/**
 * Checks whether or not the element has any children, and returns the result
 * as a boolean.
 *
 * @method hasChildren
 * @return {boolean}
 * @example&lt;animal
 * <div class='norender'><code>
 * // The following short XML file called "mammals.xml" is parsed
 * // in the code below.
 * //
 * // <?xml version="1.0"?>
 * // &lt;mammals&gt;
 * //   &lt;animal id="0" species="Capra hircus">Goat&lt;/animal&gt;
 * //   &lt;animal id="1" species="Panthera pardus">Leopard&lt;/animal&gt;
 * //   &lt;animal id="2" species="Equus zebra">Zebra&lt;/animal&gt;
 * // &lt;/mammals&gt;
 *
 * var xml;
 *
 * function preload() {
 *   xml = loadXML("assets/mammals.xml");
 * }
 *
 * function setup() {
 *   print(xml.hasChildren());
 * }
 *
 * // Sketch prints:
 * // true
 * </code></div>
 */
p5.XML.prototype.hasChildren = function() {
  return this.children.length > 0;
};

/**
 * Get the names of all of the element's children, and returns the names as an
 * array of Strings. This is the same as looping through and calling getName()
 * on each child element individually.
 *
 * @method listChildren
 * @return {Array} names of the children of the element
 * @example&lt;animal
 * <div class='norender'><code>
 * // The following short XML file called "mammals.xml" is parsed
 * // in the code below.
 * //
 * // <?xml version="1.0"?>
 * // &lt;mammals&gt;
 * //   &lt;animal id="0" species="Capra hircus">Goat&lt;/animal&gt;
 * //   &lt;animal id="1" species="Panthera pardus">Leopard&lt;/animal&gt;
 * //   &lt;animal id="2" species="Equus zebra">Zebra&lt;/animal&gt;
 * // &lt;/mammals&gt;
 *
 * var xml;
 *
 * function preload() {
 *   xml = loadXML("assets/mammals.xml");
 * }
 *
 * function setup() {
 *   print(xml.listChildren());
 * }
 *
 * // Sketch prints:
 * // ["animal", "animal", "animal"]
 * </code></div>
 */
p5.XML.prototype.listChildren = function() {
  return this.children.map(function(c) { return c.name; });
};

/**
 * Returns all of the element's children as an array of p5.XML objects. When
 * the name parameter is specified, then it will return all children that match
 * that name.
 *
 * @method getChildren
 * @param {String} [name] element name
 * @return {Array} children of the element
 * @example&lt;animal
 * <div class='norender'><code>
 * // The following short XML file called "mammals.xml" is parsed
 * // in the code below.
 * //
 * // <?xml version="1.0"?>
 * // &lt;mammals&gt;
 * //   &lt;animal id="0" species="Capra hircus">Goat&lt;/animal&gt;
 * //   &lt;animal id="1" species="Panthera pardus">Leopard&lt;/animal&gt;
 * //   &lt;animal id="2" species="Equus zebra">Zebra&lt;/animal&gt;
 * // &lt;/mammals&gt;
 *
 * var xml;
 *
 * function preload() {
 *   xml = loadXML("assets/mammals.xml");
 * }
 *
 * function setup() {
 *   var animals = xml.getChildren("animal");
 *
 *   for (var i = 0; i < animals.length; i++) {
 *     print(animals[i].getContent());
 *   }
 * }
 *
 * // Sketch prints:
 * // "Goat"
 * // "Leopard"
 * // "Zebra"
 * </code></div>
 */
p5.XML.prototype.getChildren = function(param) {
  if (param) {
    return this.children.filter(function(c) { return c.name === param; });
  }
  else {
    return this.children;
  }
};

/**
 * Returns the first of the element's children that matches the name parameter
 * or the child of the given index.It returns undefined if no matching
 * child is found.
 *
 * @method getChild
 * @param {String|Number} name element name or index
 * @return {p5.XML}
 * @example&lt;animal
 * <div class='norender'><code>
 * // The following short XML file called "mammals.xml" is parsed
 * // in the code below.
 * //
 * // <?xml version="1.0"?>
 * // &lt;mammals&gt;
 * //   &lt;animal id="0" species="Capra hircus">Goat&lt;/animal&gt;
 * //   &lt;animal id="1" species="Panthera pardus">Leopard&lt;/animal&gt;
 * //   &lt;animal id="2" species="Equus zebra">Zebra&lt;/animal&gt;
 * // &lt;/mammals&gt;
 *
 * var xml;
 *
 * function preload() {
 *   xml = loadXML("assets/mammals.xml");
 * }
 *
 * function setup() {
 *   var firstChild = xml.getChild("animal");
 *   print(firstChild.getContent());
 * }
 *
 * // Sketch prints:
 * // "Goat"
 * </code></div>
 * <div class='norender'><code>
 * var xml;
 *
 * function preload() {
 *   xml = loadXML("assets/mammals.xml");
 * }
 *
 * function setup() {
 *   var secondChild = xml.getChild(1);
 *   print(secondChild.getContent());
 * }
 *
 * // Sketch prints:
 * // "Leopard"
 * </code></div>
 */
p5.XML.prototype.getChild = function(param) {
  if(typeof param === 'string') {
    return this.children.find(function(c) {
      return c.name === param;
    });
  }
  else {
    return this.children[param];
  }
};

/**
 * Appends a new child to the element. The child can be specified with
 * either a String, which will be used as the new tag's name, or as a
 * reference to an existing p5.XML object.
 * A reference to the newly created child is returned as an p5.XML object.
 *
 * @method addChild
 * @param {Object} a p5.XML Object which will be the child to be added
 */
p5.XML.prototype.addChild = function(node) {
  if (node instanceof p5.XML) {
    this.children.push(node);
  } else {
    // PEND
  }
};

/**
 * Removes the element specified by name or index.
 *
 * @method removeChild
 * @param {String|Number} name element name or index
 * @example
 * <div class='norender'><code>
 * // The following short XML file called "mammals.xml" is parsed
 * // in the code below.
 * //
 * // <?xml version="1.0"?>
 * // &lt;mammals&gt;
 * //   &lt;animal id="0" species="Capra hircus">Goat&lt;/animal&gt;
 * //   &lt;animal id="1" species="Panthera pardus">Leopard&lt;/animal&gt;
 * //   &lt;animal id="2" species="Equus zebra">Zebra&lt;/animal&gt;
 * // &lt;/mammals&gt;
 *
 * var xml;
 *
 * function preload() {
 *   xml = loadXML("assets/mammals.xml");
 * }
 *
 * function setup() {
 *   xml.removeChild("animal");
 *   var children = xml.getChildren();
 *   for (var i=0; i<children.length; i++) {
 *     print(children[i].getContent());
 *   }
 * }
 *
 * // Sketch prints:
 * // "Leopard"
 * // "Zebra"
 * </code></div>
 * <div class='norender'><code>
 * var xml;
 *
 * function preload() {
 *   xml = loadXML("assets/mammals.xml");
 * }
 *
 * function setup() {
 *   xml.removeChild(1);
 *   var children = xml.getChildren();
 *   for (var i=0; i<children.length; i++) {
 *     print(children[i].getContent());
 *   }
 * }
 *
 * // Sketch prints:
 * // "Goat"
 * // "Zebra"
 * </code></div>
 */
p5.XML.prototype.removeChild = function(param) {
  var ind = -1;
  if(typeof param === 'string') {
    for (var i=0; i<this.children.length; i++) {
      if (this.children[i].name === param) {
        ind = i;
        break;
      }
    }
  } else {
    ind = param;
  }
  if (ind !== -1) {
    this.children.splice(ind, 1);
  }
};


/**
 * Counts the specified element's number of attributes, returned as an Number.
 *
 * @method getAttributeCount
 * @return {Number}
 * @example
 * <div class='norender'><code>
 * // The following short XML file called "mammals.xml" is parsed
 * // in the code below.
 * //
 * // <?xml version="1.0"?>
 * // &lt;mammals&gt;
 * //   &lt;animal id="0" species="Capra hircus">Goat&lt;/animal&gt;
 * //   &lt;animal id="1" species="Panthera pardus">Leopard&lt;/animal&gt;
 * //   &lt;animal id="2" species="Equus zebra">Zebra&lt;/animal&gt;
 * // &lt;/mammals&gt;
 *
 * var xml;
 *
 * function preload() {
 *   xml = loadXML("assets/mammals.xml");
 * }
 *
 * function setup() {
 *   var firstChild = xml.getChild("animal");
 *   print(firstChild.getAttributeCount());
 * }
 *
 * // Sketch prints:
 * // 2
 * </code></div>
 */
p5.XML.prototype.getAttributeCount = function() {
  return Object.keys(this.attributes).length;
};

/**
 * Gets all of the specified element's attributes, and returns them as an
 * array of Strings.
 *
 * @method listAttributes
 * @return {Array} an array of strings containing the names of attributes
 * @example
 * <div class='norender'><code>
 * // The following short XML file called "mammals.xml" is parsed
 * // in the code below.
 * //
 * // <?xml version="1.0"?>
 * // &lt;mammals&gt;
 * //   &lt;animal id="0" species="Capra hircus">Goat&lt;/animal&gt;
 * //   &lt;animal id="1" species="Panthera pardus">Leopard&lt;/animal&gt;
 * //   &lt;animal id="2" species="Equus zebra">Zebra&lt;/animal&gt;
 * // &lt;/mammals&gt;
 *
 * var xml;
 *
 * function preload() {
 *   xml = loadXML("assets/mammals.xml");
 * }
 *
 * function setup() {
 *   var firstChild = xml.getChild("animal");
 *   print(firstChild.listAttributes());
 * }
 *
 * // Sketch prints:
 * // ["id", "species"]
 * </code></div>
 */
p5.XML.prototype.listAttributes = function() {
  return Object.keys(this.attributes);
};

/**
 *  Checks whether or not an element has the specified attribute.
 *
 * @method hasAttribute
 * @param {String} the attribute to be checked
 * @return {boolean} true if attribute found else false
 * @example
 * <div class='norender'><code>
 * // The following short XML file called "mammals.xml" is parsed
 * // in the code below.
 * //
 * // <?xml version="1.0"?>
 * // &lt;mammals&gt;
 * //   &lt;animal id="0" species="Capra hircus">Goat&lt;/animal&gt;
 * //   &lt;animal id="1" species="Panthera pardus">Leopard&lt;/animal&gt;
 * //   &lt;animal id="2" species="Equus zebra">Zebra&lt;/animal&gt;
 * // &lt;/mammals&gt;
 *
 * var xml;
 *
 * function preload() {
 *   xml = loadXML("assets/mammals.xml");
 * }
 *
 * function setup() {
 *   var firstChild = xml.getChild("animal");
 *   print(firstChild.hasAttribute("species"));
 *   print(firstChild.hasAttribute("color"));
 * }
 *
 * // Sketch prints:
 * // true
 * // false
 * </code></div>
 */
p5.XML.prototype.hasAttribute = function(name) {
  return this.attributes[name] ? true : false;
};

/**
 * Returns an attribute value of the element as an Number. If the defaultValue
 * parameter is specified and the attribute doesn't exist, then defaultValue
 * is returned. If no defaultValue is specified and the attribute doesn't
 * exist, the value 0 is returned.
 *
 * @method getNumber
 * @param {String} name            the non-null full name of the attribute
 * @param {Number} [defaultValue]  the default value of the attribute
 * @return {Number}
 * @example
 * <div class='norender'><code>
 * // The following short XML file called "mammals.xml" is parsed
 * // in the code below.
 * //
 * // <?xml version="1.0"?>
 * // &lt;mammals&gt;
 * //   &lt;animal id="0" species="Capra hircus">Goat&lt;/animal&gt;
 * //   &lt;animal id="1" species="Panthera pardus">Leopard&lt;/animal&gt;
 * //   &lt;animal id="2" species="Equus zebra">Zebra&lt;/animal&gt;
 * // &lt;/mammals&gt;
 *
 * var xml;
 *
 * function preload() {
 *   xml = loadXML("assets/mammals.xml");
 * }
 *
 * function setup() {
 *   var firstChild = xml.getChild("animal");
 *   print(firstChild.getNumber("id"));
 * }
 *
 * // Sketch prints:
 * // 0
 * </code></div>
 */
p5.XML.prototype.getNumber = function(name, defaultValue) {
  return Number(this.attributes[name]) || defaultValue || 0;
};

/**
 * Returns an attribute value of the element as an String. If the defaultValue
 * parameter is specified and the attribute doesn't exist, then defaultValue
 * is returned. If no defaultValue is specified and the attribute doesn't
 * exist, null is returned.
 *
 * @method getString
 * @param {String} name            the non-null full name of the attribute
 * @param {Number} [defaultValue]  the default value of the attribute
 * @return {Number}
 * @example
 * <div class='norender'><code>
 * // The following short XML file called "mammals.xml" is parsed
 * // in the code below.
 * //
 * // <?xml version="1.0"?>
 * // &lt;mammals&gt;
 * //   &lt;animal id="0" species="Capra hircus">Goat&lt;/animal&gt;
 * //   &lt;animal id="1" species="Panthera pardus">Leopard&lt;/animal&gt;
 * //   &lt;animal id="2" species="Equus zebra">Zebra&lt;/animal&gt;
 * // &lt;/mammals&gt;
 *
 * var xml;
 *
 * function preload() {
 *   xml = loadXML("assets/mammals.xml");
 * }
 *
 * function setup() {
 *   var firstChild = xml.getChild("animal");
 *   print(firstChild.getString("species"));
 * }
 *
 * // Sketch prints:
 * // "Capra hircus"
 * </code></div>
 */
p5.XML.prototype.getString = function(name, defaultValue) {
  return String(this.attributes[name]) || defaultValue || null;
};

/**
 * Sets the content of an element's attribute. The first parameter specifies
 * the attribute name, while the second specifies the new content.
 *
 * @method setAttribute
 * @param {String} name            the full name of the attribute
 * @param {Number} value           the value of the attribute
 * @example
 * <div class='norender'><code>
 * // The following short XML file called "mammals.xml" is parsed
 * // in the code below.
 * //
 * // <?xml version="1.0"?>
 * // &lt;mammals&gt;
 * //   &lt;animal id="0" species="Capra hircus">Goat&lt;/animal&gt;
 * //   &lt;animal id="1" species="Panthera pardus">Leopard&lt;/animal&gt;
 * //   &lt;animal id="2" species="Equus zebra">Zebra&lt;/animal&gt;
 * // &lt;/mammals&gt;
 *
 * var xml;
 *
 * function preload() {
 *   xml = loadXML("assets/mammals.xml");
 * }
 *
 * function setup() {
 *   var firstChild = xml.getChild("animal");
 *   print(firstChild.getString("species"));
 *   firstChild.setAttribute("species", "Jamides zebra");
 *   print(firstChild.getString("species"));
 * }
 *
 * // Sketch prints:
 * // "Capra hircus"
 * // "Jamides zebra"
 * </code></div>
 */
p5.XML.prototype.setAttribute = function(name, value) {
  if (this.attributes[name]) {
    this.attributes[name] = value;
  }
};

/**
 * Returns the content of an element. If there is no such content,
 * defaultValue is returned if specified, otherwise null is returned.
 *
 * @method getContent
 * @param {String} [defaultValue] value returned if no content is found
 * @return {String}
 * @example
 * <div class='norender'><code>
 * // The following short XML file called "mammals.xml" is parsed
 * // in the code below.
 * //
 * // <?xml version="1.0"?>
 * // &lt;mammals&gt;
 * //   &lt;animal id="0" species="Capra hircus">Goat&lt;/animal&gt;
 * //   &lt;animal id="1" species="Panthera pardus">Leopard&lt;/animal&gt;
 * //   &lt;animal id="2" species="Equus zebra">Zebra&lt;/animal&gt;
 * // &lt;/mammals&gt;
 *
 * var xml;
 *
 * function preload() {
 *   xml = loadXML("assets/mammals.xml");
 * }
 *
 * function setup() {
 *   var firstChild = xml.getChild("animal");
 *   print(firstChild.getContent());
 * }
 *
 * // Sketch prints:
 * // "Goat"
 * </code></div>
 */
p5.XML.prototype.getContent = function(defaultValue) {
  return this.content || defaultValue || null;
};

/**
 * Sets the element's content.
 *
 * @method setContent
 * @param {String} text the new content
 * @example
 * <div class='norender'><code>
 * // The following short XML file called "mammals.xml" is parsed
 * // in the code below.
 * //
 * // <?xml version="1.0"?>
 * // &lt;mammals&gt;
 * //   &lt;animal id="0" species="Capra hircus">Goat&lt;/animal&gt;
 * //   &lt;animal id="1" species="Panthera pardus">Leopard&lt;/animal&gt;
 * //   &lt;animal id="2" species="Equus zebra">Zebra&lt;/animal&gt;
 * // &lt;/mammals&gt;
 *
 * var xml;
 *
 * function preload() {
 *   xml = loadXML("assets/mammals.xml");
 * }
 *
 * function setup() {
 *   var firstChild = xml.getChild("animal");
 *   print(firstChild.getContent());
 *   firstChild.setContent("Mountain Goat");
 *   print(firstChild.getContent());
 * }
 *
 * // Sketch prints:
 * // "Goat"
 * // "Mountain Goat"
 * </code></div>
 */
p5.XML.prototype.setContent = function( content ) {
  if(!this.children.length) {
    this.content = content;
  }
};

/* HELPERS */
/**
 * This method is called while the parsing of XML (when loadXML() is
 * called). The difference between this method and the setContent()
 * method defined later is that this one is used to set the content
 * when the node in question has more nodes under it and so on and
 * not directly text content. While in the other one is used when
 * the node in question directly has text inside it.
 *
 */
p5.XML.prototype._setCont = function(content) {
  var str;
  str = content;
  str = str.replace(/\s\s+/g, ',');
  //str = str.split(',');
  this.content = str;
};

/**
 * This method is called while the parsing of XML (when loadXML() is
 * called). The XML node is passed and its attributes are stored in the
 * p5.XML's attribute Object.
 *
 */
p5.XML.prototype._setAttributes = function(node) {
  var  i, att = {};
  for( i = 0; i < node.attributes.length; i++) {
    att[node.attributes[i].nodeName] = node.attributes[i].nodeValue;
  }
  this.attributes = att;
};

module.exports = p5.XML;
},{"../core/core":37}],63:[function(_dereq_,module,exports){
/**
 * @module Math
 * @submodule Calculation
 * @for p5
 * @requires core
 */

'use strict';

var p5 = _dereq_('../core/core');

/**
 * Calculates the absolute value (magnitude) of a number. Maps to Math.abs().
 * The absolute value of a number is always positive.
 *
 * @method abs
 * @param  {Number} n number to compute
 * @return {Number}   absolute value of given number
 * @example
 * <div class = "norender"><code>
 * function setup() {
 *   var x = -3;
 *   var y = abs(x);
 *
 *   print(x); // -3
 *   print(y); // 3
 * }
 * </code></div>
 */
p5.prototype.abs = Math.abs;

/**
 * Calculates the closest int value that is greater than or equal to the
 * value of the parameter. Maps to Math.ceil(). For example, ceil(9.03)
 * returns the value 10.
 *
 * @method ceil
 * @param  {Number} n number to round up
 * @return {Number}   rounded up number
 * @example
 * <div><code>
 * function draw() {
 *   background(200);
 *   // map, mouseX between 0 and 5.
 *   var ax = map(mouseX, 0, 100, 0, 5);
 *   var ay = 66;
 *
 *   //Get the ceiling of the mapped number.
 *   var bx = ceil(map(mouseX, 0, 100, 0,5));
 *   var by = 33;
 *
 *   // Multiply the mapped numbers by 20 to more easily
 *   // see the changes.
 *   stroke(0);
 *   fill(0);
 *   line(0, ay, ax * 20, ay);
 *   line(0, by, bx * 20, by);
 *
 *   // Reformat the float returned by map and draw it.
 *   noStroke();
 *   text(nfc(ax, 2,2), ax, ay - 5);
 *   text(nfc(bx,1,1), bx, by - 5);
 * }
 * </code></div>
 */
p5.prototype.ceil = Math.ceil;

/**
 * Constrains a value between a minimum and maximum value.
 *
 * @method constrain
 * @param  {Number} n    number to constrain
 * @param  {Number} low  minimum limit
 * @param  {Number} high maximum limit
 * @return {Number}      constrained number
 * @example
 * <div><code>
 * function draw() {
 *   background(200);
 *
 *   var leftWall = 25;
 *   var rightWall = 75;
 *
 *   // xm is just the mouseX, while
 *   // xc is the mouseX, but constrained
 *   // between the leftWall and rightWall!
 *   var xm = mouseX;
 *   var xc = constrain(mouseX, leftWall, rightWall);
 *
 *   // Draw the walls.
 *   stroke(150);
 *   line(leftWall, 0, leftWall, height);
 *   line(rightWall, 0, rightWall, height);
 *
 *   // Draw xm and xc as circles.
 *   noStroke();
 *   fill(150);
 *   ellipse(xm, 33, 9,9); // Not Constrained
 *   fill(0);
 *   ellipse(xc, 66, 9,9); // Constrained
 * }
 * </code></div>
 */
p5.prototype.constrain = function(n, low, high) {
  return Math.max(Math.min(n, high), low);
};

/**
 * Calculates the distance between two points.
 *
 * @method dist
 * @param  {Number} x1 x-coordinate of the first point
 * @param  {Number} y1 y-coordinate of the first point
 * @param  {Number} [z1] z-coordinate of the first point
 * @param  {Number} x2 x-coordinate of the second point
 * @param  {Number} y2 y-coordinate of the second point
 * @param  {Number} [z2] z-coordinate of the second point
 * @return {Number}    distance between the two points
 * @example
 * <div><code>
 * // Move your mouse inside the canvas to see the
 * // change in distance between two points!
 * function draw() {
 *   background(200);
 *   fill(0);
 *
 *   var x1 = 10;
 *   var y1 = 90;
 *   var x2 = mouseX;
 *   var y2 = mouseY;
 *
 *   line(x1, y1, x2, y2);
 *   ellipse(x1, y1, 7, 7);
 *   ellipse(x2, y2, 7, 7);
 *
 *   // d is the length of the line
 *   // the distance from point 1 to point 2.
 *   var d = int(dist(x1, y1, x2, y2));
 *
 *   // Let's write d along the line we are drawing!
 *   push();
 *   translate( (x1+x2)/2, (y1+y2)/2 );
 *   rotate( atan2(y2-y1,x2-x1) );
 *   text(nfc(d,1,1), 0, -5);
 *   pop();
 *   // Fancy!
 * }
 * </code></div>
 */
p5.prototype.dist = function(x1, y1, z1, x2, y2, z2) {
  if (arguments.length === 4) {
    // In the case of 2d: z1 means x2 and x2 means y2
    return Math.sqrt( (z1-x1)*(z1-x1) + (x2-y1)*(x2-y1) );
  } else if (arguments.length === 6) {
    return Math.sqrt( (x2-x1)*(x2-x1) + (y2-y1)*(y2-y1) + (z2-z1)*(z2-z1) );
  }
};

/**
 * Returns Euler's number e (2.71828...) raised to the power of the n
 * parameter. Maps to Math.exp().
 *
 * @method exp
 * @param  {Number} n exponent to raise
 * @return {Number}   e^n
 * @example
 * <div><code>
 * function draw() {
 *   background(200);
 *
 *   // Compute the exp() function with a value between 0 and 2
 *   var xValue = map(mouseX, 0, width, 0, 2);
 *   var yValue = exp(xValue);
 *
 *   var y = map(yValue, 0, 8, height, 0);
 *
 *   var legend = "exp (" + nfc(xValue, 3) +")\n= " + nf(yValue, 1, 4);
 *   stroke(150);
 *   line(mouseX, y, mouseX, height);
 *   fill(0);
 *   text(legend, 5, 15);
 *   noStroke();
 *   ellipse (mouseX,y, 7, 7);
 *
 *   // Draw the exp(x) curve,
 *   // over the domain of x from 0 to 2
 *   noFill();
 *   stroke(0);
 *   beginShape();
 *   for (var x = 0; x < width; x++) {
 *     xValue = map(x, 0, width, 0, 2);
 *     yValue = exp(xValue);
 *     y = map(yValue, 0, 8, height, 0);
 *     vertex(x, y);
 *   }
 *
 *   endShape();
 *   line(0, 0, 0, height);
 *   line(0, height-1, width, height-1);
 * }
 * </code></div>
 */
p5.prototype.exp = Math.exp;

/**
 * Calculates the closest int value that is less than or equal to the
 * value of the parameter. Maps to Math.floor().
 *
 * @method floor
 * @param  {Number} n number to round down
 * @return {Number}   rounded down number
 * @example
 * <div><code>
 * function draw() {
 *   background(200);
 *   //map, mouseX between 0 and 5.
 *   var ax = map(mouseX, 0, 100, 0, 5);
 *   var ay = 66;
 *
 *   //Get the floor of the mapped number.
 *   var bx = floor(map(mouseX, 0, 100, 0,5));
 *   var by = 33;
 *
 *   // Multiply the mapped numbers by 20 to more easily
 *   // see the changes.
 *   stroke(0);
 *   fill(0);
 *   line(0, ay, ax * 20, ay);
 *   line(0, by, bx * 20, by);
 *
 *   // Reformat the float returned by map and draw it.
 *   noStroke();
 *   text(nfc(ax, 2,2), ax, ay - 5);
 *   text(nfc(bx,1,1), bx, by - 5);
 * }
 * </code></div>
 */
p5.prototype.floor = Math.floor;

/**
 * Calculates a number between two numbers at a specific increment. The amt
 * parameter is the amount to interpolate between the two values where 0.0
 * equal to the first point, 0.1 is very near the first point, 0.5 is
 * half-way in between, etc. The lerp function is convenient for creating
 * motion along a straight path and for drawing dotted lines.
 *
 * @method lerp
 * @param  {Number} start first value
 * @param  {Number} stop  second value
 * @param  {Number} amt   number between 0.0 and 1.0
 * @return {Number}       lerped value
 * @example
 * <div><code>
 * function setup() {
 *   background(200);
 *   var a = 20;
 *   var b = 80;
 *   var c = lerp(a,b, .2);
 *   var d = lerp(a,b, .5);
 *   var e = lerp(a,b, .8);
 *
 *   var y = 50
 *
 *   strokeWeight(5);
 *   stroke(0); // Draw the original points in black
 *   point(a, y);
 *   point(b, y);
 *
 *   stroke(100); // Draw the lerp points in gray
 *   point(c, y);
 *   point(d, y);
 *   point(e, y);
 * }
 * </code></div>
 */
p5.prototype.lerp = function(start, stop, amt) {
  return amt*(stop-start)+start;
};

/**
 * Calculates the natural logarithm (the base-e logarithm) of a number. This
 * function expects the n parameter to be a value greater than 0.0. Maps to
 * Math.log().
 *
 * @method log
 * @param  {Number} n number greater than 0
 * @return {Number}   natural logarithm of n
 * @example
 * <div><code>
 * function draw() {
 *   background(200);
 *   var maxX = 2.8;
 *   var maxY = 1.5;
 *
 *   // Compute the natural log of a value between 0 and maxX
 *   var xValue = map(mouseX, 0, width, 0, maxX);
 *   if (xValue > 0) { // Cannot take the log of a negative number.
 *     var yValue = log(xValue);
 *     var y = map(yValue, -maxY, maxY, height, 0);
 *
 *     // Display the calculation occurring.
 *     var legend = "log(" + nf(xValue, 1, 2) + ")\n= " + nf(yValue, 1, 3);
 *     stroke(150);
 *     line(mouseX, y, mouseX, height);
 *     fill(0);
 *     text (legend, 5, 15);
 *     noStroke();
 *     ellipse (mouseX, y, 7, 7);
 *   }
 *
 *   // Draw the log(x) curve,
 *   // over the domain of x from 0 to maxX
 *   noFill();
 *   stroke(0);
 *   beginShape();
 *   for(var x=0; x < width; x++) {
 *     xValue = map(x, 0, width, 0, maxX);
 *     yValue = log(xValue);
 *     y = map(yValue, -maxY, maxY, height, 0);
 *     vertex(x, y);
 *   }
 *   endShape();
 *   line(0,0,0,height);
 *   line(0,height/2,width, height/2);
 * }
 * </code></div>
 */
p5.prototype.log = Math.log;

/**
 * Calculates the magnitude (or length) of a vector. A vector is a direction
 * in space commonly used in computer graphics and linear algebra. Because it
 * has no "start" position, the magnitude of a vector can be thought of as
 * the distance from the coordinate 0,0 to its x,y value. Therefore, mag() is
 * a shortcut for writing dist(0, 0, x, y).
 *
 * @method mag
 * @param  {Number} a first value
 * @param  {Number} b second value
 * @return {Number}   magnitude of vector from (0,0) to (a,b)
 * @example
 * <div><code>
 * function setup() {
 *   var x1 = 20;
 *   var x2 = 80;
 *   var y1 = 30;
 *   var y2 = 70;
 *
 *   line(0, 0, x1, y1);
 *   print(mag(x1, y1));  // Prints "36.05551"
 *   line(0, 0, x2, y1);
 *   print(mag(x2, y1));  // Prints "85.44004"
 *   line(0, 0, x1, y2);
 *   print(mag(x1, y2));  // Prints "72.8011"
 *   line(0, 0, x2, y2);
 *   print(mag(x2, y2));  // Prints "106.30146"
 * }
 * </code></div>
 */
p5.prototype.mag = function(x, y) {
  return Math.sqrt(x*x+y*y);
};

/**
 * Re-maps a number from one range to another.
 * <br><br>
 * In the first example above, the number 25 is converted from a value in the
 * range of 0 to 100 into a value that ranges from the left edge of the
 * window (0) to the right edge (width).
 *
 * @method map
 * @param  {Number} value  the incoming value to be converted
 * @param  {Number} start1 lower bound of the value's current range
 * @param  {Number} stop1  upper bound of the value's current range
 * @param  {Number} start2 lower bound of the value's target range
 * @param  {Number} stop2  upper bound of the value's target range
 * @return {Number}        remapped number
 * @example
 *   <div><code>
 *     var value = 25;
 *     var m = map(value, 0, 100, 0, width);
 *     ellipse(m, 50, 10, 10);
 *   </code></div>
 *
 *   <div><code>
 *     function setup() {
 *       noStroke();
 *     }
 *
 *     function draw() {
 *       background(204);
 *       var x1 = map(mouseX, 0, width, 25, 75);
 *       ellipse(x1, 25, 25, 25);
 *       var x2 = map(mouseX, 0, width, 0, 100);
 *       ellipse(x2, 75, 25, 25);
 *     }
 *   </code></div>
 */
p5.prototype.map = function(n, start1, stop1, start2, stop2) {
  return ((n-start1)/(stop1-start1))*(stop2-start2)+start2;
};

/**
 * Determines the largest value in a sequence of numbers, and then returns
 * that value. max() accepts any number of Number parameters, or an Array
 * of any length.
 *
 * @method max
 * @param  {Number|Array} n0 Numbers to compare
 * @return {Number}          maximum Number
 * @example
 * <div><code>
 * function setup() {
 *   // Change the elements in the array and run the sketch
 *   // to show how max() works!
 *   numArray = new Array(2,1,5,4,8,9);
 *   fill(0);
 *   noStroke();
 *   text("Array Elements", 0, 10);
 *   // Draw all numbers in the array
 *   var spacing = 15;
 *   var elemsY = 25;
 *   for(var i = 0; i < numArray.length; i++) {
 *     text(numArray[i], i * spacing, elemsY);
 *   }
 *   maxX = 33;
 *   maxY = 80;
 *   // Draw the Maximum value in the array.
 *   textSize(32);
 *   text(max(numArray), maxX, maxY);
 * }
 * </code></div>
 */
p5.prototype.max = function() {
  if (arguments[0] instanceof Array) {
    return Math.max.apply(null,arguments[0]);
  } else {
    return Math.max.apply(null,arguments);
  }
};

/**
 * Determines the smallest value in a sequence of numbers, and then returns
 * that value. min() accepts any number of Number parameters, or an Array
 * of any length.
 *
 * @method min
 * @param  {Number|Array} n0 Numbers to compare
 * @return {Number}          minimum Number
 * @example
 * <div><code>
 * function setup() {
 *   // Change the elements in the array and run the sketch
 *   // to show how min() works!
 *   numArray = new Array(2,1,5,4,8,9);
 *   fill(0);
 *   noStroke();
 *   text("Array Elements", 0, 10);
 *   // Draw all numbers in the array
 *   var spacing = 15;
 *   var elemsY = 25;
 *   for(var i = 0; i < numArray.length; i++) {
 *     text(numArray[i], i * spacing, elemsY);
 *   }
 *   maxX = 33;
 *   maxY = 80;
 *   // Draw the Minimum value in the array.
 *   textSize(32);
 *   text(min(numArray), maxX, maxY);
 * }
 * </code></div>
 */
p5.prototype.min = function() {
  if (arguments[0] instanceof Array) {
    return Math.min.apply(null,arguments[0]);
  } else {
    return Math.min.apply(null,arguments);
  }
};

/**
 * Normalizes a number from another range into a value between 0 and 1.
 * Identical to map(value, low, high, 0, 1).
 * Numbers outside of the range are not clamped to 0 and 1, because
 * out-of-range values are often intentional and useful. (See the second
 * example above.)
 *
 * @method norm
 * @param  {Number} value incoming value to be normalized
 * @param  {Number} start lower bound of the value's current range
 * @param  {Number} stop  upper bound of the value's current range
 * @return {Number}       normalized number
 * @example
 * <div><code>
 * function draw() {
 *   background(200);
 *   currentNum = mouseX;
 *   lowerBound = 0;
 *   upperBound = width; //100;
 *   normalized = norm(currentNum, lowerBound, upperBound);
 *   lineY = 70
 *   line(0, lineY, width, lineY);
 *   //Draw an ellipse mapped to the non-normalized value.
 *   noStroke();
 *   fill(50)
 *   var s = 7; // ellipse size
 *   ellipse(currentNum, lineY, s, s);
 *
 *   // Draw the guide
 *   guideY = lineY + 15;
 *   text("0", 0, guideY);
 *   textAlign(RIGHT);
 *   text("100", width, guideY);
 *
 *   // Draw the normalized value
 *   textAlign(LEFT);
 *   fill(0);
 *   textSize(32);
 *   normalY = 40;
 *   normalX = 20;
 *   text(normalized, normalX, normalY);
 * }
 * </code></div>
 */
p5.prototype.norm = function(n, start, stop) {
  return this.map(n, start, stop, 0, 1);
};

/**
 * Facilitates exponential expressions. The pow() function is an efficient
 * way of multiplying numbers by themselves (or their reciprocals) in large
 * quantities. For example, pow(3, 5) is equivalent to the expression
 * 3*3*3*3*3 and pow(3, -5) is equivalent to 1 / 3*3*3*3*3. Maps to
 * Math.pow().
 *
 * @method pow
 * @param  {Number} n base of the exponential expression
 * @param  {Number} e power by which to raise the base
 * @return {Number}   n^e
 * @example
 * <div><code>
 * function setup() {
 *   //Exponentially increase the size of an ellipse.
 *   eSize = 3; // Original Size
 *   eLoc = 10; // Original Location
 *
 *   ellipse(eLoc, eLoc, eSize, eSize);
 *
 *   ellipse(eLoc*2, eLoc*2, pow(eSize, 2), pow(eSize, 2));
 *
 *   ellipse(eLoc*4, eLoc*4, pow(eSize, 3), pow(eSize, 3));
 *
 *   ellipse(eLoc*8, eLoc*8, pow(eSize, 4), pow(eSize, 4));
 * }
 * </code></div>
 */
p5.prototype.pow = Math.pow;

/**
 * Calculates the integer closest to the n parameter. For example,
 * round(133.8) returns the value 134. Maps to Math.round().
 *
 * @method round
 * @param  {Number} n number to round
 * @return {Number}   rounded number
 * @example
 * <div><code>
 * function draw() {
 *   background(200);
 *   //map, mouseX between 0 and 5.
 *   var ax = map(mouseX, 0, 100, 0, 5);
 *   var ay = 66;
 *
 *   // Round the mapped number.
 *   var bx = round(map(mouseX, 0, 100, 0,5));
 *   var by = 33;
 *
 *   // Multiply the mapped numbers by 20 to more easily
 *   // see the changes.
 *   stroke(0);
 *   fill(0);
 *   line(0, ay, ax * 20, ay);
 *   line(0, by, bx * 20, by);
 *
 *   // Reformat the float returned by map and draw it.
 *   noStroke();
 *   text(nfc(ax, 2,2), ax, ay - 5);
 *   text(nfc(bx,1,1), bx, by - 5);
 * }
 * </code></div>
 */
p5.prototype.round = Math.round;

/**
 * Squares a number (multiplies a number by itself). The result is always a
 * positive number, as multiplying two negative numbers always yields a
 * positive result. For example, -1 * -1 = 1.
 *
 * @method sq
 * @param  {Number} n number to square
 * @return {Number}   squared number
 * @example
 * <div><code>
 * function draw() {
 *   background(200);
 *   eSize = 7;
 *   x1 = map(mouseX, 0, width, 0, 10);
 *   y1 = 80;
 *   x2 = sq(x1);
 *   y2 = 20;
 *
 *   // Draw the non-squared.
 *   line(0, y1, width, y1);
 *   ellipse(x1, y1, eSize, eSize);
 *
 *   // Draw the squared.
 *   line(0, y2, width, y2);
 *   ellipse(x2, y2, eSize, eSize);
 *
 *   // Draw dividing line.
 *   stroke(100)
 *   line(0, height/2, width, height/2);
 *
 *   // Draw text.
 *   var spacing = 15;
 *   noStroke();
 *   fill(0);
 *   text("x = " + x1, 0, y1 + spacing);
 *   text("sq(x) = " + x2, 0, y2 + spacing);
 * }
 * </code></div>
 */
p5.prototype.sq = function(n) { return n*n; };

/**
 * Calculates the square root of a number. The square root of a number is
 * always positive, even though there may be a valid negative root. The
 * square root s of number a is such that s*s = a. It is the opposite of
 * squaring. Maps to Math.sqrt().
 *
 * @method sqrt
 * @param  {Number} n non-negative number to square root
 * @return {Number}   square root of number
 * @example
 * <div><code>
 * function draw() {
 *   background(200);
 *   eSize = 7;
 *   x1 = mouseX;
 *   y1 = 80;
 *   x2 = sqrt(x1);
 *   y2 = 20;
 *
 *   // Draw the non-squared.
 *   line(0, y1, width, y1);
 *   ellipse(x1, y1, eSize, eSize);
 *
 *   // Draw the squared.
 *   line(0, y2, width, y2);
 *   ellipse(x2, y2, eSize, eSize);
 *
 *   // Draw dividing line.
 *   stroke(100)
 *   line(0, height/2, width, height/2);
 *
 *   // Draw text.
 *   noStroke();
 *   fill(0);
 *   var spacing = 15;
 *   text("x = " + x1, 0, y1 + spacing);
 *   text("sqrt(x) = " + x2, 0, y2 + spacing);
 * }
 * </code></div>
 */
p5.prototype.sqrt = Math.sqrt;

module.exports = p5;

},{"../core/core":37}],64:[function(_dereq_,module,exports){
/**
 * @module Math
 * @submodule Math
 * @for p5
 * @requires core
 */

'use strict';

var p5 = _dereq_('../core/core');


/**
 * Creates a new p5.Vector (the datatype for storing vectors). This provides a
 * two or three dimensional vector, specifically a Euclidean (also known as
 * geometric) vector. A vector is an entity that has both magnitude and
 * direction.
 *
 * @method createVector
 * @param {Number} [x] x component of the vector
 * @param {Number} [y] y component of the vector
 * @param {Number} [z] z component of the vector
 */
p5.prototype.createVector = function (x, y, z) {
  if (this instanceof p5) {
    return new p5.Vector(this, arguments);
  } else {
    return new p5.Vector(x, y, z);
  }
};

module.exports = p5;

},{"../core/core":37}],65:[function(_dereq_,module,exports){
//////////////////////////////////////////////////////////////

// http://mrl.nyu.edu/~perlin/noise/
// Adapting from PApplet.java
// which was adapted from toxi
// which was adapted from the german demo group farbrausch
// as used in their demo "art": http://www.farb-rausch.de/fr010src.zip

// someday we might consider using "improved noise"
// http://mrl.nyu.edu/~perlin/paper445.pdf
// See: https://github.com/shiffman/The-Nature-of-Code-Examples-p5.js/
//      blob/master/introduction/Noise1D/noise.js

/**
 * @module Math
 * @submodule Noise
 * @for p5
 * @requires core
 */

'use strict';

var p5 = _dereq_('../core/core');

var PERLIN_YWRAPB = 4;
var PERLIN_YWRAP = 1<<PERLIN_YWRAPB;
var PERLIN_ZWRAPB = 8;
var PERLIN_ZWRAP = 1<<PERLIN_ZWRAPB;
var PERLIN_SIZE = 4095;

var perlin_octaves = 4; // default to medium smooth
var perlin_amp_falloff = 0.5; // 50% reduction/octave

var scaled_cosine = function(i) {
  return 0.5*(1.0-Math.cos(i*Math.PI));
};

var perlin; // will be initialized lazily by noise() or noiseSeed()


/**
 * Returns the Perlin noise value at specified coordinates. Perlin noise is
 * a random sequence generator producing a more natural ordered, harmonic
 * succession of numbers compared to the standard <b>random()</b> function.
 * It was invented by Ken Perlin in the 1980s and been used since in
 * graphical applications to produce procedural textures, natural motion,
 * shapes, terrains etc.<br /><br /> The main difference to the
 * <b>random()</b> function is that Perlin noise is defined in an infinite
 * n-dimensional space where each pair of coordinates corresponds to a
 * fixed semi-random value (fixed only for the lifespan of the program; see
 * the noiseSeed() function). p5.js can compute 1D, 2D and 3D noise,
 * depending on the number of coordinates given. The resulting value will
 * always be between 0.0 and 1.0. The noise value can be animated by moving
 * through the noise space as demonstrated in the example above. The 2nd
 * and 3rd dimension can also be interpreted as time.<br /><br />The actual
 * noise is structured similar to an audio signal, in respect to the
 * function's use of frequencies. Similar to the concept of harmonics in
 * physics, perlin noise is computed over several octaves which are added
 * together for the final result. <br /><br />Another way to adjust the
 * character of the resulting sequence is the scale of the input
 * coordinates. As the function works within an infinite space the value of
 * the coordinates doesn't matter as such, only the distance between
 * successive coordinates does (eg. when using <b>noise()</b> within a
 * loop). As a general rule the smaller the difference between coordinates,
 * the smoother the resulting noise sequence will be. Steps of 0.005-0.03
 * work best for most applications, but this will differ depending on use.
 *
 *
 * @method noise
 * @param  {Number} x   x-coordinate in noise space
 * @param  {Number} y   y-coordinate in noise space
 * @param  {Number} z   z-coordinate in noise space
 * @return {Number}     Perlin noise value (between 0 and 1) at specified
 *                      coordinates
 * @example
 * <div>
 * <code>var xoff = 0.0;
 *
 * function draw() {
 *   background(204);
 *   xoff = xoff + .01;
 *   var n = noise(xoff) * width;
 *   line(n, 0, n, height);
 * }
 * </code>
 * </div>
 * <div>
 * <code>var noiseScale=0.02;
 *
 * function draw() {
 *   background(0);
 *   for (var x=0; x < width; x++) {
 *     var noiseVal = noise((mouseX+x)*noiseScale, mouseY*noiseScale);
 *     stroke(noiseVal*255);
 *     line(x, mouseY+noiseVal*80, x, height);
 *   }
 * }
 * </code>
 * </div>
 */
p5.prototype.noise = function(x,y,z) {
  y = y || 0;
  z = z || 0;

  if (perlin == null) {
    perlin = new Array(PERLIN_SIZE + 1);
    for (var i = 0; i < PERLIN_SIZE + 1; i++) {
      perlin[i] = Math.random();
    }
  }

  if (x<0) { x=-x; }
  if (y<0) { y=-y; }
  if (z<0) { z=-z; }

  var xi=Math.floor(x), yi=Math.floor(y), zi=Math.floor(z);
  var xf = x - xi;
  var yf = y - yi;
  var zf = z - zi;
  var rxf, ryf;

  var r=0;
  var ampl=0.5;

  var n1,n2,n3;

  for (var o=0; o<perlin_octaves; o++) {
    var of=xi+(yi<<PERLIN_YWRAPB)+(zi<<PERLIN_ZWRAPB);

    rxf = scaled_cosine(xf);
    ryf = scaled_cosine(yf);

    n1  = perlin[of&PERLIN_SIZE];
    n1 += rxf*(perlin[(of+1)&PERLIN_SIZE]-n1);
    n2  = perlin[(of+PERLIN_YWRAP)&PERLIN_SIZE];
    n2 += rxf*(perlin[(of+PERLIN_YWRAP+1)&PERLIN_SIZE]-n2);
    n1 += ryf*(n2-n1);

    of += PERLIN_ZWRAP;
    n2  = perlin[of&PERLIN_SIZE];
    n2 += rxf*(perlin[(of+1)&PERLIN_SIZE]-n2);
    n3  = perlin[(of+PERLIN_YWRAP)&PERLIN_SIZE];
    n3 += rxf*(perlin[(of+PERLIN_YWRAP+1)&PERLIN_SIZE]-n3);
    n2 += ryf*(n3-n2);

    n1 += scaled_cosine(zf)*(n2-n1);

    r += n1*ampl;
    ampl *= perlin_amp_falloff;
    xi<<=1;
    xf*=2;
    yi<<=1;
    yf*=2;
    zi<<=1;
    zf*=2;

    if (xf>=1.0) { xi++; xf--; }
    if (yf>=1.0) { yi++; yf--; }
    if (zf>=1.0) { zi++; zf--; }
  }
  return r;
};


/**
 *
 * Adjusts the character and level of detail produced by the Perlin noise
 * function. Similar to harmonics in physics, noise is computed over
 * several octaves. Lower octaves contribute more to the output signal and
 * as such define the overall intensity of the noise, whereas higher octaves
 * create finer grained details in the noise sequence.
 * <br><br>
 * By default, noise is computed over 4 octaves with each octave contributing
 * exactly half than its predecessor, starting at 50% strength for the 1st
 * octave. This falloff amount can be changed by adding an additional function
 * parameter. Eg. a falloff factor of 0.75 means each octave will now have
 * 75% impact (25% less) of the previous lower octave. Any value between
 * 0.0 and 1.0 is valid, however note that values greater than 0.5 might
 * result in greater than 1.0 values returned by <b>noise()</b>.
 * <br><br>
 * By changing these parameters, the signal created by the <b>noise()</b>
 * function can be adapted to fit very specific needs and characteristics.
 *
 * @method noiseDetail
 * @param {Number} lod number of octaves to be used by the noise
 * @param {Number} falloff falloff factor for each octave
 * @example
 * <div>
 * <code>
 *
 * var noiseVal;
 * var noiseScale=0.02;
 *
 * function setup() {
 *   createCanvas(100,100);
 * }
 *
 * function draw() {
 *   background(0);
 *   for (var y = 0; y < height; y++) {
 *     for (var x = 0; x < width/2; x++) {
 *       noiseDetail(2,0.2);
 *       noiseVal = noise((mouseX+x) * noiseScale,
 *                        (mouseY+y) * noiseScale);
 *       stroke(noiseVal*255);
 *       point(x,y);
 *       noiseDetail(8,0.65);
 *       noiseVal = noise((mouseX + x + width/2) * noiseScale,
 *                        (mouseY + y) * noiseScale);
 *       stroke(noiseVal*255);
 *       point(x + width/2, y);
 *     }
 *   }
 * }
 * </code>
 * </div>
 */
p5.prototype.noiseDetail = function(lod, falloff) {
  if (lod>0)     { perlin_octaves=lod; }
  if (falloff>0) { perlin_amp_falloff=falloff; }
};

/**
 * Sets the seed value for <b>noise()</b>. By default, <b>noise()</b>
 * produces different results each time the program is run. Set the
 * <b>value</b> parameter to a constant to return the same pseudo-random
 * numbers each time the software is run.
 *
 * @method noiseSeed
 * @param {Number} seed   the seed value
 * @example
 * <div>
 * <code>var xoff = 0.0;
 *
 * function setup() {
 *   noiseSeed(99);
 *   stroke(0, 10);
 * }
 *
 * function draw() {
 *   xoff = xoff + .01;
 *   var n = noise(xoff) * width;
 *   line(n, 0, n, height);
 * }
 * </code>
 * </div>
 */
p5.prototype.noiseSeed = function(seed) {
  // Linear Congruential Generator
  // Variant of a Lehman Generator
  var lcg = (function() {
    // Set to values from http://en.wikipedia.org/wiki/Numerical_Recipes
    // m is basically chosen to be large (as it is the max period)
    // and for its relationships to a and c
    var m = 4294967296,
    // a - 1 should be divisible by m's prime factors
    a = 1664525,
     // c and m should be co-prime
    c = 1013904223,
    seed, z;
    return {
      setSeed : function(val) {
        // pick a random seed if val is undefined or null
        // the >>> 0 casts the seed to an unsigned 32-bit integer
        z = seed = (val == null ? Math.random() * m : val) >>> 0;
      },
      getSeed : function() {
        return seed;
      },
      rand : function() {
        // define the recurrence relationship
        z = (a * z + c) % m;
        // return a float in [0, 1)
        // if z = m then z / m = 0 therefore (z % m) / m < 1 always
        return z / m;
      }
    };
  }());

  lcg.setSeed(seed);
  perlin = new Array(PERLIN_SIZE + 1);
  for (var i = 0; i < PERLIN_SIZE + 1; i++) {
    perlin[i] = lcg.rand();
  }
};

module.exports = p5;

},{"../core/core":37}],66:[function(_dereq_,module,exports){
/**
 * @module Math
 * @submodule Math
 * @requires constants
 */

'use strict';

var p5 = _dereq_('../core/core');
var polarGeometry = _dereq_('./polargeometry');
var constants = _dereq_('../core/constants');

/**
 * A class to describe a two or three dimensional vector, specifically
 * a Euclidean (also known as geometric) vector. A vector is an entity
 * that has both magnitude and direction. The datatype, however, stores
 * the components of the vector (x, y for 2D, and x, y, z for 3D). The magnitude
 * and direction can be accessed via the methods mag() and heading().
 * <br><br>
 * In many of the p5.js examples, you will see p5.Vector used to describe a
 * position, velocity, or acceleration. For example, if you consider a rectangle
 * moving across the screen, at any given instant it has a position (a vector
 * that points from the origin to its location), a velocity (the rate at which
 * the object's position changes per time unit, expressed as a vector), and
 * acceleration (the rate at which the object's velocity changes per time
 * unit, expressed as a vector).
 * <br><br>
 * Since vectors represent groupings of values, we cannot simply use
 * traditional addition/multiplication/etc. Instead, we'll need to do some
 * "vector" math, which is made easy by the methods inside the p5.Vector class.
 *
 * @class p5.Vector
 * @constructor
 * @param {Number} [x] x component of the vector
 * @param {Number} [y] y component of the vector
 * @param {Number} [z] z component of the vector
 * @example
 * <div>
 * <code>
 * var v1 = createVector(40, 50);
 * var v2 = createVector(40, 50);
 *
 * ellipse(v1.x, v1.y, 50, 50);
 * ellipse(v2.x, v2.y, 50, 50);
 * v1.add(v2);
 * ellipse(v1.x, v1.y, 50, 50);
 * </code>
 * </div>
 */
p5.Vector = function() {
  var x,y,z;
  // This is how it comes in with createVector()
  if(arguments[0] instanceof p5) {
    // save reference to p5 if passed in
    this.p5 = arguments[0];
    x  = arguments[1][0] || 0;
    y  = arguments[1][1] || 0;
    z  = arguments[1][2] || 0;
  // This is what we'll get with new p5.Vector()
  } else {
    x = arguments[0] || 0;
    y = arguments[1] || 0;
    z = arguments[2] || 0;
  }
  /**
   * The x component of the vector
   * @property x
   * @type {Number}
   */
  this.x = x;
  /**
   * The y component of the vector
   * @property y
   * @type {Number}
   */
  this.y = y;
  /**
   * The z component of the vector
   * @property z
   * @type {Number}
   */
  this.z = z;
};

/**
 * Returns a string representation of a vector v by calling String(v)
 * or v.toString(). This method is useful for logging vectors in the
 * console.
 * @method  toString
 * @example
 * <div class = "norender"><code>
 * function setup() {
 *   var v = createVector(20,30);
 *   print(String(v)); // prints "p5.Vector Object : [20, 30, 0]"
 * }
 * </div></code>
 *
 */
p5.Vector.prototype.toString = function p5VectorToString() {
  return 'p5.Vector Object : ['+ this.x +', '+ this.y +', '+ this.z + ']';
};

/**
 * Sets the x, y, and z component of the vector using two or three separate
 * variables, the data from a p5.Vector, or the values from a float array.
 * @method set
 *
 * @param {Number|p5.Vector|Array} [x] the x component of the vector or a
 *                                     p5.Vector or an Array
 * @param {Number}                 [y] the y component of the vector
 * @param {Number}                 [z] the z component of the vector
 * @example
 * <div class="norender">
 * <code>
 * function setup() {
 *    var v = createVector(1, 2, 3);
 *    v.set(4,5,6); // Sets vector to [4, 5, 6]
 *
 *    var v1 = createVector(0, 0, 0);
 *    var arr = [1, 2, 3];
 *    v1.set(arr); // Sets vector to [1, 2, 3]
 * }
 * </code>
 * </div>
 */
p5.Vector.prototype.set = function (x, y, z) {
  if (x instanceof p5.Vector) {
    this.x = x.x || 0;
    this.y = x.y || 0;
    this.z = x.z || 0;
    return this;
  }
  if (x instanceof Array) {
    this.x = x[0] || 0;
    this.y = x[1] || 0;
    this.z = x[2] || 0;
    return this;
  }
  this.x = x || 0;
  this.y = y || 0;
  this.z = z || 0;
  return this;
};

/**
 * Gets a copy of the vector, returns a p5.Vector object.
 *
 * @method copy
 * @return {p5.Vector} the copy of the p5.Vector object
 * @example
 * <div class="norender">
 * <code>
 * var v1 = createVector(1, 2, 3);
 * var v2 = v1.copy();
 * print(v1.x == v2.x && v1.y == v2.y && v1.z == v2.z);
 * // Prints "true"
 * </code>
 * </div>
 */
p5.Vector.prototype.copy = function () {
  if (this.p5) {
    return new p5.Vector(this.p5,[this.x, this.y, this.z]);
  } else {
    return new p5.Vector(this.x,this.y,this.z);
  }
};

/**
 * Adds x, y, and z components to a vector, adds one vector to another, or
 * adds two independent vectors together. The version of the method that adds
 * two vectors together is a static method and returns a p5.Vector, the others
 * acts directly on the vector. See the examples for more context.
 *
 * @method add
 * @chainable
 * @param  {Number|p5.Vector|Array} x   the x component of the vector to be
 *                                      added or a p5.Vector or an Array
 * @param  {Number}                 [y] the y component of the vector to be
 *                                      added
 * @param  {Number}                 [z] the z component of the vector to be
 *                                      added
 * @return {p5.Vector}                  the p5.Vector object.
 * @example
 * <div class="norender">
 * <code>
 * var v = createVector(1, 2, 3);
 * v.add(4,5,6);
 * // v's compnents are set to [5, 7, 9]
 * </code>
 * </div>
 * <div class="norender">
 * <code>
 * // Static method
 * var v1 = createVector(1, 2, 3);
 * var v2 = createVector(2, 3, 4);
 *
 * var v3 = p5.Vector.add(v1, v2);
 * // v3 has components [3, 5, 7]
 * </code>
 * </div>
 */
p5.Vector.prototype.add = function (x, y, z) {
  if (x instanceof p5.Vector) {
    this.x += x.x || 0;
    this.y += x.y || 0;
    this.z += x.z || 0;
    return this;
  }
  if (x instanceof Array) {
    this.x += x[0] || 0;
    this.y += x[1] || 0;
    this.z += x[2] || 0;
    return this;
  }
  this.x += x || 0;
  this.y += y || 0;
  this.z += z || 0;
  return this;
};

/**
 * Subtracts x, y, and z components from a vector, subtracts one vector from
 * another, or subtracts two independent vectors. The version of the method
 * that subtracts two vectors is a static method and returns a p5.Vector, the
 * other acts directly on the vector. See the examples for more context.
 *
 * @method sub
 * @chainable
 * @param  {Number|p5.Vector|Array} x   the x component of the vector or a
 *                                      p5.Vector or an Array
 * @param  {Number}                 [y] the y component of the vector
 * @param  {Number}                 [z] the z component of the vector
 * @return {p5.Vector}                  p5.Vector object.
 * @example
 * <div class="norender">
 * <code>
 * var v = createVector(4, 5, 6);
 * v.sub(1, 1, 1);
 * // v's compnents are set to [3, 4, 5]
 * </code>
 * </div>
 *
 * <div class="norender">
 * <code>
 * // Static method
 * var v1 = createVector(2, 3, 4);
 * var v2 = createVector(1, 2, 3);
 *
 * var v3 = p5.Vector.sub(v1, v2);
 * // v3 has compnents [1, 1, 1]
 * </code>
 * </div>
 */
p5.Vector.prototype.sub = function (x, y, z) {
  if (x instanceof p5.Vector) {
    this.x -= x.x || 0;
    this.y -= x.y || 0;
    this.z -= x.z || 0;
    return this;
  }
  if (x instanceof Array) {
    this.x -= x[0] || 0;
    this.y -= x[1] || 0;
    this.z -= x[2] || 0;
    return this;
  }
  this.x -= x || 0;
  this.y -= y || 0;
  this.z -= z || 0;
  return this;
};

/**
 * Multiply the vector by a scalar. The static version of this method
 * creates a new p5.Vector while the non static version acts on the vector
 * directly. See the examples for more context.
 *
 * @method mult
 * @chainable
 * @param  {Number}    n the number to multiply with the vector
 * @return {p5.Vector} a reference to the p5.Vector object (allow chaining)
 * @example
 * <div class="norender">
 * <code>
 * var v = createVector(1, 2, 3);
 * v.mult(2);
 * // v's compnents are set to [2, 4, 6]
 * </code>
 * </div>
 *
 * <div class="norender">
 * <code>
 * // Static method
 * var v1 = createVector(1, 2, 3);
 * var v2 = p5.Vector.mult(v1, 2);
 * // v2 has compnents [2, 4, 6]
 * </code>
 * </div>
 */
p5.Vector.prototype.mult = function (n) {
  this.x *= n || 0;
  this.y *= n || 0;
  this.z *= n || 0;
  return this;
};

/**
 * Divide the vector by a scalar. The static version of this method creates a
 * new p5.Vector while the non static version acts on the vector directly.
 * See the examples for more context.
 *
 * @method div
 * @chainable
 * @param  {number}    n the number to divide the vector by
 * @return {p5.Vector} a reference to the p5.Vector object (allow chaining)
 * @example
 * <div class="norender">
 * <code>
 * var v = createVector(6, 4, 2);
 * v.div(2); //v's compnents are set to [3, 2, 1]
 * </code>
 * </div>
 *
 * <div class="norender">
 * <code>
 * // Static method
 * var v1  = createVector(6, 4, 2);
 * var v2 = p5.Vector.div(v, 2);
 * // v2 has compnents [3, 2, 1]
 * </code>
 * </div>
 */
p5.Vector.prototype.div = function (n) {
  this.x /= n;
  this.y /= n;
  this.z /= n;
  return this;
};

/**
 * Calculates the magnitude (length) of the vector and returns the result as
 * a float (this is simply the equation sqrt(x*x + y*y + z*z).)
 *
 * @method mag
 * @return {Number} magnitude of the vector
 * @example
 * <div class="norender">
 * <code>
 * var v = createVector(20.0, 30.0, 40.0);
 * var m = v.mag();
 * print(m); // Prints "53.85164807134504"
 * </code>
 * </div>
 */
p5.Vector.prototype.mag = function () {
  return Math.sqrt(this.magSq());
};

/**
 * Calculates the squared magnitude of the vector and returns the result
 * as a float (this is simply the equation <em>(x*x + y*y + z*z)</em>.)
 * Faster if the real length is not required in the
 * case of comparing vectors, etc.
 *
 * @method magSq
 * @return {number} squared magnitude of the vector
 * @example
 * <div class="norender">
 * <code>
 * // Static method
 * var v1 = createVector(6, 4, 2);
 * print(v1.magSq()); // Prints "56"
 * </code>
 * </div>
 */
p5.Vector.prototype.magSq = function () {
  var x = this.x, y = this.y, z = this.z;
  return (x * x + y * y + z * z);
};

/**
 * Calculates the dot product of two vectors. The version of the method
 * that computes the dot product of two independent vectors is a static
 * method. See the examples for more context.
 *
 *
 * @method dot
 * @param  {Number|p5.Vector} x   x component of the vector or a p5.Vector
 * @param  {Number}           [y] y component of the vector
 * @param  {Number}           [z] z component of the vector
 * @return {Number}                 the dot product
 *
 * @example
 * <div class="norender">
 * <code>
 * var v1 = createVector(1, 2, 3);
 * var v2 = createVector(2, 3, 4);
 *
 * print(v1.dot(v2)); // Prints "20"
 * </code>
 * </div>
 *
 * <div class="norender">
 * <code>
 * //Static method
 * var v1 = createVector(1, 2, 3);
 * var v2 = createVector(3, 2, 1);
 * print (p5.Vector.dot(v1, v2)); // Prints "10"
 * </code>
 * </div>
 */
p5.Vector.prototype.dot = function (x, y, z) {
  if (x instanceof p5.Vector) {
    return this.dot(x.x, x.y, x.z);
  }
  return this.x * (x || 0) +
         this.y * (y || 0) +
         this.z * (z || 0);
};

/**
 * Calculates and returns a vector composed of the cross product between
 * two vectors. Both the static and non static methods return a new p5.Vector.
 * See the examples for more context.
 *
 * @method cross
 * @param  {p5.Vector} v p5.Vector to be crossed
 * @return {p5.Vector}   p5.Vector composed of cross product
 * @example
 * <div class="norender">
 * <code>
 * var v1 = createVector(1, 2, 3);
 * var v2 = createVector(1, 2, 3);
 *
 * v1.cross(v2); // v's components are [0, 0, 0]
 * </code>
 * </div>
 *
 * <div class="norender">
 * <code>
 * // Static method
 * var v1 = createVector(1, 0, 0);
 * var v2 = createVector(0, 1, 0);
 *
 * var crossProduct = p5.Vector.cross(v1, v2);
 * // crossProduct has components [0, 0, 1]
 * </code>
 * </div>
 */
p5.Vector.prototype.cross = function (v) {
  var x = this.y * v.z - this.z * v.y;
  var y = this.z * v.x - this.x * v.z;
  var z = this.x * v.y - this.y * v.x;
  if (this.p5) {
    return new p5.Vector(this.p5,[x,y,z]);
  } else {
    return new p5.Vector(x,y,z);
  }
};

/**
 * Calculates the Euclidean distance between two points (considering a
 * point as a vector object).
 *
 * @method dist
 * @param  {p5.Vector} v the x, y, and z coordinates of a p5.Vector
 * @return {Number}      the distance
 * @example
 * <div class="norender">
 * <code>
 * var v1 = createVector(1, 0, 0);
 * var v2 = createVector(0, 1, 0);
 *
 * var distance = v1.dist(v2); // distance is 1.4142...
 * </code>
 * </div>
 * <div class="norender">
 * <code>
 * // Static method
 * var v1 = createVector(1, 0, 0);
 * var v2 = createVector(0, 1, 0);
 *
 * var distance = p5.Vector.dist(v1,v2);
 * // distance is 1.4142...
 * </code>
 * </div>
 */
p5.Vector.prototype.dist = function (v) {
  var d = v.copy().sub(this);
  return d.mag();
};

/**
 * Normalize the vector to length 1 (make it a unit vector).
 *
 * @method normalize
 * @return {p5.Vector} normalized p5.Vector
 * @example
 * <div class="norender">
 * <code>
 * var v = createVector(10, 20, 2);
 * // v has compnents [10.0, 20.0, 2.0]
 * v.normalize();
 * // v's compnents are set to
 * // [0.4454354, 0.8908708, 0.089087084]
 * </code>
 * </div>
 *
 */
p5.Vector.prototype.normalize = function () {
  return this.div(this.mag());
};

/**
 * Limit the magnitude of this vector to the value used for the <b>max</b>
 * parameter.
 *
 * @method limit
 * @param  {Number}    max the maximum magnitude for the vector
 * @return {p5.Vector}     the modified p5.Vector
 * @example
 * <div class="norender">
 * <code>
 * var v = createVector(10, 20, 2);
 * // v has compnents [10.0, 20.0, 2.0]
 * v.limit(5);
 * // v's compnents are set to
 * // [2.2271771, 4.4543543, 0.4454354]
 * </code>
 * </div>
 */
p5.Vector.prototype.limit = function (l) {
  var mSq = this.magSq();
  if(mSq > l*l) {
    this.div(Math.sqrt(mSq)); //normalize it
    this.mult(l);
  }
  return this;
};

/**
 * Set the magnitude of this vector to the value used for the <b>len</b>
 * parameter.
 *
 * @method setMag
 * @param  {number}    len the new length for this vector
 * @return {p5.Vector}     the modified p5.Vector
 * @example
 * <div class="norender">
 * <code>
 * var v1 = createVector(10, 20, 2);
 * // v has compnents [10.0, 20.0, 2.0]
 * v1.setMag(10);
 * // v's compnents are set to [6.0, 8.0, 0.0]
 * </code>
 * </div>
 */
p5.Vector.prototype.setMag = function (n) {
  return this.normalize().mult(n);
};

/**
 * Calculate the angle of rotation for this vector (only 2D vectors)
 *
 * @method heading
 * @return {Number} the angle of rotation
 * @example
 * <div class = "norender"><code>
 * function setup() {
 *   var v1 = createVector(30,50);
 *   print(v1.heading()); // 1.0303768265243125
 *
 *   var v1 = createVector(40,50);
 *   print(v1.heading()); // 0.8960553845713439
 *
 *   var v1 = createVector(30,70);
 *   print(v1.heading()); // 1.1659045405098132
 * }
 * </div></code>
 */
p5.Vector.prototype.heading = function () {
  var h = Math.atan2(this.y, this.x);
  if (this.p5) {
    if (this.p5._angleMode === constants.RADIANS) {
      return h;
    } else {
      return polarGeometry.radiansToDegrees(h);
    }
  } else {
    return h;
  }
};

/**
 * Rotate the vector by an angle (only 2D vectors), magnitude remains the
 * same
 *
 * @method rotate
 * @param  {number}    angle the angle of rotation
 * @return {p5.Vector} the modified p5.Vector
 * @example
 * <div class="norender">
 * <code>
 * var v = createVector(10.0, 20.0);
 * // v has compnents [10.0, 20.0, 0.0]
 * v.rotate(HALF_PI);
 * // v's compnents are set to [-20.0, 9.999999, 0.0]
 * </code>
 * </div>
 */
p5.Vector.prototype.rotate = function (a) {
  if (this.p5) {
    if (this.p5._angleMode === constants.DEGREES) {
      a = polarGeometry.degreesToRadians(a);
    }
  }
  var newHeading = this.heading() + a;
  var mag = this.mag();
  this.x = Math.cos(newHeading) * mag;
  this.y = Math.sin(newHeading) * mag;
  return this;
};

/**
 * Linear interpolate the vector to another vector
 *
 * @method lerp
 * @param  {p5.Vector} x   the x component or the p5.Vector to lerp to
 * @param  {p5.Vector} [y] y the y component
 * @param  {p5.Vector} [z] z the z component
 * @param  {Number}    amt the amount of interpolation; some value between 0.0
 *                         (old vector) and 1.0 (new vector). 0.1 is very near
 *                         the new vector. 0.5 is halfway in between.
 * @return {p5.Vector}     the modified p5.Vector
 * @example
 * <div class="norender">
 * <code>
 * var v = createVector(1, 1, 0);
 *
 * v.lerp(3, 3, 0, 0.5); // v now has components [2,2,0]
 * </code>
 * </div>
 *
 * <div class="norender">
 * <code>
 * var v1 = createVector(0, 0, 0);
 * var v2 = createVector(100, 100, 0);
 *
 * var v3 = p5.Vector.lerp(v1, v2, 0.5);
 * // v3 has components [50,50,0]
 * </code>
 * </div>
 */
p5.Vector.prototype.lerp = function (x, y, z, amt) {
  if (x instanceof p5.Vector) {
    return this.lerp(x.x, x.y, x.z, y);
  }
  this.x += (x - this.x) * amt || 0;
  this.y += (y - this.y) * amt || 0;
  this.z += (z - this.z) * amt || 0;
  return this;
};

/**
 * Return a representation of this vector as a float array. This is only
 * for temporary use. If used in any other fashion, the contents should be
 * copied by using the <b>p5.Vector.copy()</b> method to copy into your own
 * array.
 *
 * @method array
 * @return {Array} an Array with the 3 values
 * @example
 * <div class = "norender"><code>
 * function setup() {
 *   var v = createVector(20,30);
 *   print(v.array()); // Prints : Array [20, 30, 0]
 * }
 * </div></code>
 * <div class="norender">
 * <code>
 * var v = createVector(10.0, 20.0, 30.0);
 * var f = v.array();
 * print(f[0]); // Prints "10.0"
 * print(f[1]); // Prints "20.0"
 * print(f[2]); // Prints "30.0"
 * </code>
 * </div>
 */
p5.Vector.prototype.array = function () {
  return [this.x || 0, this.y || 0, this.z || 0];
};

/**
 * Equality check against a p5.Vector
 *
 * @method equals
 * @param {Number|p5.Vector|Array} [x] the x component of the vector or a
 *                                     p5.Vector or an Array
 * @param {Number}                 [y] the y component of the vector
 * @param {Number}                 [z] the z component of the vector
 * @return {Boolean} whether the vectors are equals
 * @example
 * <div class = "norender"><code>
 * v1 = createVector(5,10,20);
 * v2 = createVector(5,10,20);
 * v3 = createVector(13,10,19);
 *
 * print(v1.equals(v2.x,v2.y,v2.z)); // true
 * print(v1.equals(v3.x,v3.y,v3.z)); // false
 * </div></code>
 * <div class="norender">
 * <code>
 * var v1 = createVector(10.0, 20.0, 30.0);
 * var v2 = createVector(10.0, 20.0, 30.0);
 * var v3 = createVector(0.0, 0.0, 0.0);
 * print (v1.equals(v2)) // true
 * print (v1.equals(v3)) // false
 * </code>
 * </div>
 */
p5.Vector.prototype.equals = function (x, y, z) {
  var a, b, c;
  if (x instanceof p5.Vector) {
    a = x.x || 0;
    b = x.y || 0;
    c = x.z || 0;
  } else if (x instanceof Array) {
    a = x[0] || 0;
    b = x[1] || 0;
    c = x[2] || 0;
  } else {
    a = x || 0;
    b = y || 0;
    c = z || 0;
  }
  return this.x === a && this.y === b && this.z === c;
};


// Static Methods


/**
 * Make a new 2D unit vector from an angle
 *
 * @method fromAngle
 * @static
 * @param {Number}     angle the desired angle
 * @return {p5.Vector}       the new p5.Vector object
 * @example
 * <div>
 * <code>
 * function draw() {
 *   background (200);
 *
 *   // Create a variable, proportional to the mouseX,
 *   // varying from 0-360, to represent an angle in degrees.
 *   angleMode(DEGREES);
 *   var myDegrees = map(mouseX, 0,width, 0,360);
 *
 *   // Display that variable in an onscreen text.
 *   // (Note the nfc() function to truncate additional decimal places,
 *   // and the "\xB0" character for the degree symbol.)
 *   var readout = "angle = " + nfc(myDegrees,1,1) + "\xB0"
 *   noStroke();
 *   fill (0);
 *   text (readout, 5, 15);
 *
 *   // Create a p5.Vector using the fromAngle function,
 *   // and extract its x and y components.
 *   var v = p5.Vector.fromAngle(radians(myDegrees));
 *   var vx = v.x;
 *   var vy = v.y;
 *
 *   push();
 *   translate (width/2, height/2);
 *   noFill();
 *   stroke (150);
 *   line (0,0, 30,0);
 *   stroke (0);
 *   line (0,0, 30*vx, 30*vy);
 *   pop()
 * }
 * </code>
 * </div>
 */
p5.Vector.fromAngle = function(angle) {
  if (this.p5) {
    if (this.p5._angleMode === constants.DEGREES) {
      angle = polarGeometry.degreesToRadians(angle);
    }
  }
  if (this.p5) {
    return new p5.Vector(this.p5,[Math.cos(angle),Math.sin(angle),0]);
  } else {
    return new p5.Vector(Math.cos(angle),Math.sin(angle),0);
  }
};

/**
 * Make a new 2D unit vector from a random angle
 *
 * @method random2D
 * @static
 * @return {p5.Vector} the new p5.Vector object
 * @example
 * <div class="norender">
 * <code>
 * var v = p5.Vector.random2D();
 * // May make v's attributes something like:
 * // [0.61554617, -0.51195765, 0.0] or
 * // [-0.4695841, -0.14366731, 0.0] or
 * // [0.6091097, -0.22805278, 0.0]
 * </code>
 * </div>
 */
p5.Vector.random2D = function () {
  var angle;
  // A lot of nonsense to determine if we know about a
  // p5 sketch and whether we should make a random angle in degrees or radians
  if (this.p5) {
    if (this.p5._angleMode === constants.DEGREES) {
      angle = this.p5.random(360);
    } else {
      angle = this.p5.random(constants.TWO_PI);
    }
  } else {
    angle = Math.random()*Math.PI*2;
  }
  return this.fromAngle(angle);
};

/**
 * Make a new random 3D unit vector.
 *
 * @method random3D
 * @static
 * @return {p5.Vector} the new p5.Vector object
 * @example
 * <div class="norender">
 * <code>
 * var v = p5.Vector.random3D();
 * // May make v's attributes something like:
 * // [0.61554617, -0.51195765, 0.599168] or
 * // [-0.4695841, -0.14366731, -0.8711202] or
 * // [0.6091097, -0.22805278, -0.7595902]
 * </code>
 * </div>
 */
p5.Vector.random3D = function () {
  var angle,vz;
  // If we know about p5
  if (this.p5) {
    angle = this.p5.random(0,constants.TWO_PI);
    vz = this.p5.random(-1,1);
  } else {
    angle = Math.random()*Math.PI*2;
    vz = Math.random()*2-1;
  }
  var vx = Math.sqrt(1-vz*vz)*Math.cos(angle);
  var vy = Math.sqrt(1-vz*vz)*Math.sin(angle);
  if (this.p5) {
    return new p5.Vector(this.p5,[vx,vy,vz]);
  } else {
    return new p5.Vector(vx,vy,vz);
  }
};


/**
 * Adds two vectors together and returns a new one.
 *
 * @static
 * @param  {p5.Vector} v1 a p5.Vector to add
 * @param  {p5.Vector} v2 a p5.Vector to add
 * @param  {p5.Vector} target if undefined a new vector will be created
 * @return {p5.Vector} the resulting p5.Vector
 *
 */

p5.Vector.add = function (v1, v2, target) {
  if (!target) {
    target = v1.copy();
  } else {
    target.set(v1);
  }
  target.add(v2);
  return target;
};

/**
 * Subtracts one p5.Vector from another and returns a new one.  The second
 * vector (v2) is subtracted from the first (v1), resulting in v1-v2.
 *
 * @static
 * @param  {p5.Vector} v1 a p5.Vector to subtract from
 * @param  {p5.Vector} v2 a p5.Vector to subtract
 * @param  {p5.Vector} target if undefined a new vector will be created
 * @return {p5.Vector} the resulting p5.Vector
 */

p5.Vector.sub = function (v1, v2, target) {
  if (!target) {
    target = v1.copy();
  } else {
    target.set(v1);
  }
  target.sub(v2);
  return target;
};


/**
 * Multiplies a vector by a scalar and returns a new vector.
 *
 * @static
 * @param  {p5.Vector} v the p5.Vector to multiply
 * @param  {Number}  n the scalar
 * @param  {p5.Vector} target if undefined a new vector will be created
 * @return {p5.Vector}  the resulting new p5.Vector
 */
p5.Vector.mult = function (v, n, target) {
  if (!target) {
    target = v.copy();
  } else {
    target.set(v);
  }
  target.mult(n);
  return target;
};

/**
 * Divides a vector by a scalar and returns a new vector.
 *
 * @static
 * @param  {p5.Vector} v the p5.Vector to divide
 * @param  {Number}  n the scalar
 * @param  {p5.Vector} target if undefined a new vector will be created
 * @return {p5.Vector} the resulting new p5.Vector
 */
p5.Vector.div = function (v, n, target) {
  if (!target) {
    target = v.copy();
  } else {
    target.set(v);
  }
  target.div(n);
  return target;
};


/**
 * Calculates the dot product of two vectors.
 *
 * @static
 * @param  {p5.Vector} v1 the first p5.Vector
 * @param  {p5.Vector} v2 the second p5.Vector
 * @return {Number}     the dot product
 */
p5.Vector.dot = function (v1, v2) {
  return v1.dot(v2);
};

/**
 * Calculates the cross product of two vectors.
 *
 * @static
 * @param  {p5.Vector} v1 the first p5.Vector
 * @param  {p5.Vector} v2 the second p5.Vector
 * @return {Number}     the cross product
 */
p5.Vector.cross = function (v1, v2) {
  return v1.cross(v2);
};

/**
 * Calculates the Euclidean distance between two points (considering a
 * point as a vector object).
 *
 * @static
 * @param  {p5.Vector} v1 the first p5.Vector
 * @param  {p5.Vector} v2 the second p5.Vector
 * @return {Number}     the distance
 */
p5.Vector.dist = function (v1,v2) {
  return v1.dist(v2);
};

/**
 * Linear interpolate a vector to another vector and return the result as a
 * new vector.
 *
 * @static
 * @param {p5.Vector} v1 a starting p5.Vector
 * @param {p5.Vector} v2 the p5.Vector to lerp to
 * @param {Number}       the amount of interpolation; some value between 0.0
 *                       (old vector) and 1.0 (new vector). 0.1 is very near
 *                       the new vector. 0.5 is halfway in between.
 */
p5.Vector.lerp = function (v1, v2, amt, target) {
  if (!target) {
    target = v1.copy();
  } else {
    target.set(v1);
  }
  target.lerp(v2, amt);
  return target;
};

/**
 * Calculates and returns the angle (in radians) between two vectors.
 * @method angleBetween
 * @static
 * @param  {p5.Vector} v1 the x, y, and z components of a p5.Vector
 * @param  {p5.Vector} v2 the x, y, and z components of a p5.Vector
 * @return {Number}       the angle between (in radians)
 * @example
 * <div class="norender">
 * <code>
 * var v1 = createVector(1, 0, 0);
 * var v2 = createVector(0, 1, 0);
 *
 * var angle = p5.Vector.angleBetween(v1, v2);
 * // angle is PI/2
 * </code>
 * </div>
 */
p5.Vector.angleBetween = function (v1, v2) {
  var angle = Math.acos(v1.dot(v2) / (v1.mag() * v2.mag()));
  if (this.p5) {
    if (this.p5._angleMode === constants.DEGREES) {
      angle = polarGeometry.radiansToDegrees(angle);
    }
  }
  return angle;
};

/**
 * @static
 */
p5.Vector.mag = function (vecT){
  var x = vecT.x,
    y = vecT.y,
    z = vecT.z;
  var magSq = x * x + y * y + z * z;
  return Math.sqrt(magSq);
};

module.exports = p5.Vector;

},{"../core/constants":36,"../core/core":37,"./polargeometry":67}],67:[function(_dereq_,module,exports){

module.exports = {

  degreesToRadians: function(x) {
    return 2 * Math.PI * x / 360;
  },

  radiansToDegrees: function(x) {
    return 360 * x / (2 * Math.PI);
  }

};

},{}],68:[function(_dereq_,module,exports){
/**
 * @module Math
 * @submodule Random
 * @for p5
 * @requires core
 */

'use strict';

var p5 = _dereq_('../core/core');

var seeded = false;

// Linear Congruential Generator
// Variant of a Lehman Generator
var lcg = (function() {
  // Set to values from http://en.wikipedia.org/wiki/Numerical_Recipes
  // m is basically chosen to be large (as it is the max period)
  // and for its relationships to a and c
  var m = 4294967296,
    // a - 1 should be divisible by m's prime factors
    a = 1664525,
    // c and m should be co-prime
    c = 1013904223,
    seed, z;
  return {
    setSeed : function(val) {
      // pick a random seed if val is undefined or null
      // the >>> 0 casts the seed to an unsigned 32-bit integer
      z = seed = (val == null ? Math.random() * m : val) >>> 0;
    },
    getSeed : function() {
      return seed;
    },
    rand : function() {
      // define the recurrence relationship
      z = (a * z + c) % m;
      // return a float in [0, 1)
      // if z = m then z / m = 0 therefore (z % m) / m < 1 always
      return z / m;
    }
  };
}());

/**
 * Sets the seed value for random().
 *
 * By default, random() produces different results each time the program
 * is run. Set the seed parameter to a constant to return the same
 * pseudo-random numbers each time the software is run.
 *
 * @method randomSeed
 * @param {Number} seed   the seed value
 * @example
 * <div>
 * <code>
 * randomSeed(99);
 * for (var i=0; i < 100; i++) {
 *   var r = random(0, 255);
 *   stroke(r);
 *   line(i, 0, i, 100);
 * }
 * </code>
 * </div>
 */
p5.prototype.randomSeed = function(seed) {
  lcg.setSeed(seed);
  seeded = true;
};

/**
 * Return a random floating-point number.
 *
 * Takes either 0, 1 or 2 arguments.
 *
 * If no argument is given, returns a random number from 0
 * up to (but not including) 1.
 *
 * If one argument is given and it is a number, returns a random number from 0
 * up to (but not including) the number.
 *
 * If one argument is given and it is an array, returns a random element from
 * that array.
 *
 * If two arguments are given, returns a random number from the
 * first argument up to (but not including) the second argument.
 *
 * @method random
 * @param  {Number} [min]   the lower bound (inclusive)
 * @param  {Number} [max]   the upper bound (exclusive)
 * @return {Number|mixed} the random number or a random element in choices
 * @example
 * <div>
 * <code>
 * for (var i = 0; i < 100; i++) {
 *   var r = random(50);
 *   stroke(r*5);
 *   line(50, i, 50+r, i);
 * }
 * </code>
 * </div>
 * <div>
 * <code>
 * for (var i = 0; i < 100; i++) {
 *   var r = random(-50, 50);
 *   line(50,i,50+r,i);
 * }
 * </code>
 * </div>
 * <div>
 * <code>
 * // Get a random element from an array using the random(Array) syntax
 * var words = [ "apple", "bear", "cat", "dog" ];
 * var word = random(words);  // select random word
 * text(word,10,50);  // draw the word
 * </code>
 * </div>
 */
/**
 * @method random
 * @param  {Array} choices   the array to choose from
 * @return {mixed} the random element from the array
 * @example
 */
p5.prototype.random = function (min, max) {

  var rand;

  if (seeded) {
    rand  = lcg.rand();
  } else {
    rand = Math.random();
  }

  if (arguments.length === 0) {
    return rand;
  } else
  if (arguments.length === 1) {
    if (arguments[0] instanceof Array) {
      return arguments[0][Math.floor(rand * arguments[0].length)];
    } else {
      return rand * min;
    }
  } else {
    if (min > max) {
      var tmp = min;
      min = max;
      max = tmp;
    }

    return rand * (max-min) + min;
  }
};


/**
 *
 * Returns a random number fitting a Gaussian, or
 * normal, distribution. There is theoretically no minimum or maximum
 * value that randomGaussian() might return. Rather, there is
 * just a very low probability that values far from the mean will be
 * returned; and a higher probability that numbers near the mean will
 * be returned.
 * <br><br>
 * Takes either 0, 1 or 2 arguments.<br>
 * If no args, returns a mean of 0 and standard deviation of 1.<br>
 * If one arg, that arg is the mean (standard deviation is 1).<br>
 * If two args, first is mean, second is standard deviation.
 *
 * @method randomGaussian
 * @param  {Number} mean  the mean
 * @param  {Number} sd    the standard deviation
 * @return {Number} the random number
 * @example
 * <div>
 * <code>for (var y = 0; y < 100; y++) {
 *  var x = randomGaussian(50,15);
 *  line(50, y, x, y);
 *}
 * </code>
 * </div>
 * <div>
 * <code>
 *var distribution = new Array(360);
 *
 *function setup() {
 *  createCanvas(100, 100);
 *  for (var i = 0; i < distribution.length; i++) {
 *    distribution[i] = floor(randomGaussian(0,15));
 *  }
 *}
 *
 *function draw() {
 *  background(204);
 *
 *  translate(width/2, width/2);
 *
 *  for (var i = 0; i < distribution.length; i++) {
 *    rotate(TWO_PI/distribution.length);
 *    stroke(0);
 *    var dist = abs(distribution[i]);
 *    line(0, 0, dist, 0);
 *  }
 *}
 * </code>
 * </div>
 */
var y2;
var previous = false;
p5.prototype.randomGaussian = function(mean, sd)  {
  var y1,x1,x2,w;
  if (previous) {
    y1 = y2;
    previous = false;
  } else {
    do {
      x1 = this.random(2) - 1;
      x2 = this.random(2) - 1;
      w = x1 * x1 + x2 * x2;
    } while (w >= 1);
    w = Math.sqrt((-2 * Math.log(w))/w);
    y1 = x1 * w;
    y2 = x2 * w;
    previous = true;
  }

  var m = mean || 0;
  var s = sd || 1;
  return y1*s + m;
};

module.exports = p5;

},{"../core/core":37}],69:[function(_dereq_,module,exports){
/**
 * @module Math
 * @submodule Trigonometry
 * @for p5
 * @requires core
 * @requires polargeometry
 * @requires constants
 */

'use strict';

var p5 = _dereq_('../core/core');
var polarGeometry = _dereq_('./polargeometry');
var constants = _dereq_('../core/constants');

p5.prototype._angleMode = constants.RADIANS;

/**
 * The inverse of cos(), returns the arc cosine of a value. This function
 * expects the values in the range of -1 to 1 and values are returned in
 * the range 0 to PI (3.1415927).
 *
 * @method acos
 * @param  {Number} value the value whose arc cosine is to be returned
 * @return {Number}       the arc cosine of the given value
 *
 * @example
 * <div class= “norender">
 * <code>
 * var a = PI;
 * var c = cos(a);
 * var ac = acos(c);
 * // Prints: "3.1415927 : -1.0 : 3.1415927"
 * println(a + " : " + c + " : " +  ac);
 * </code>
 * </div>
 *
 * <div class= “norender">
 * <code>
 * var a = PI + PI/4.0;
 * var c = cos(a);
 * var ac = acos(c);
 * // Prints: "3.926991 : -0.70710665 : 2.3561943"
 * println(a + " : " + c + " : " +  ac);
 * </code>
 * </div>
 */
p5.prototype.acos = function(ratio) {
  if (this._angleMode === constants.RADIANS) {
    return Math.acos(ratio);
  } else {
    return polarGeometry.radiansToDegrees(Math.acos(ratio));
  }
};

/**
 * The inverse of sin(), returns the arc sine of a value. This function
 * expects the values in the range of -1 to 1 and values are returned
 * in the range -PI/2 to PI/2.
 *
 * @method asin
 * @param  {Number} value the value whose arc sine is to be returned
 * @return {Number}       the arc sine of the given value
 *
 * @example
 * <div class= “norender">
 * <code>
 * var a = PI + PI/3;
 * var s = sin(a);
 * var as = asin(s);
 * // Prints: "1.0471976 : 0.86602545 : 1.0471976"
 * println(a + " : " + s + " : " +  as);
 * </code>
 * </div>
 *
 * <div class= “norender">
 * <code>
 * var a = PI + PI/3.0;
 * var s = sin(a);
 * var as = asin(s);
 * // Prints: "4.1887903 : -0.86602545 : -1.0471976"
 * println(a + " : " + s + " : " +  as);
 * </code>
 * </div>
 *
 */
p5.prototype.asin = function(ratio) {
  if (this._angleMode === constants.RADIANS) {
    return Math.asin(ratio);
  } else {
    return polarGeometry.radiansToDegrees(Math.asin(ratio));
  }
};

/**
 * The inverse of tan(), returns the arc tangent of a value. This function
 * expects the values in the range of -Infinity to Infinity (exclusive) and
 * values are returned in the range -PI/2 to PI/2.
 *
 * @method atan
 * @param  {Number} value the value whose arc tangent is to be returned
 * @return {Number}       the arc tangent of the given value
 *
 * @example
 * <div class= “norender">
 * <code>
 * var a = PI + PI/3;
 * var t = tan(a);
 * var at = atan(t);
 * // Prints: "1.0471976 : 1.7320509 : 1.0471976"
 * println(a + " : " + t + " : " +  at);
 * </code>
 * </div>
 *
 * <div class= “norender">
 * <code>
 * var a = PI + PI/3.0;
 * var t = tan(a);
 * var at = atan(t);
 * // Prints: "4.1887903 : 1.7320513 : 1.0471977"
 * println(a + " : " + t + " : " +  at);
 * </code>
 * </div>
 *
 */
p5.prototype.atan = function(ratio) {
  if (this._angleMode === constants.RADIANS) {
    return Math.atan(ratio);
  } else {
    return polarGeometry.radiansToDegrees(Math.atan(ratio));
  }
};

/**
 * Calculates the angle (in radians) from a specified point to the coordinate
 * origin as measured from the positive x-axis. Values are returned as a
 * float in the range from PI to -PI. The atan2() function is most often used
 * for orienting geometry to the position of the cursor.
 * <br><br>
 * Note: The y-coordinate of the point is the first parameter, and the
 * x-coordinate is the second parameter, due the the structure of calculating
 * the tangent.
 *
 * @method atan2
 * @param  {Number} y y-coordinate of the point
 * @param  {Number} x x-coordinate of the point
 * @return {Number}   the arc tangent of the given point
 *
 * @example
 * <div>
 * <code>
 * function draw() {
 *   background(204);
 *   translate(width/2, height/2);
 *   var a = atan2(mouseY-height/2, mouseX-width/2);
 *   rotate(a);
 *   rect(-30, -5, 60, 10);
 * }
 * </code>
 * </div>
 */
p5.prototype.atan2 = function (y, x) {
  if (this._angleMode === constants.RADIANS) {
    return Math.atan2(y, x);
  } else {
    return polarGeometry.radiansToDegrees(Math.atan2(y, x));
  }
};

/**
 * Calculates the cosine of an angle. This function takes into account the
 * current angleMode. Values are returned in the range -1 to 1.
 *
 * @method cos
 * @param  {Number} angle the angle
 * @return {Number}       the cosine of the angle
 *
 * @example
 * <div>
 * <code>
 * var a = 0.0;
 * var inc = TWO_PI/25.0;
 * for (var i = 0; i < 25; i++) {
 *   line(i*4, 50, i*4, 50+cos(a)*40.0);
 *   a = a + inc;
 * }
 * </code>
 * </div>
 *
 */
p5.prototype.cos = function(angle) {
  if (this._angleMode === constants.RADIANS) {
    return Math.cos(angle);
  } else {
    return Math.cos(this.radians(angle));
  }
};

/**
 * Calculates the sine of an angle. This function takes into account the
 * current angleMode. Values are returned in the range -1 to 1.
 *
 * @method sin
 * @param  {Number} angle the angle
 * @return {Number}       the sine of the angle
 *
 * @example
 * <div>
 * <code>
 * var a = 0.0;
 * var inc = TWO_PI/25.0;
 * for (var i = 0; i < 25; i++) {
 *   line(i*4, 50, i*4, 50+sin(a)*40.0);
 *   a = a + inc;
 * }
 * </code>
 * </div>
 */
p5.prototype.sin = function(angle) {
  if (this._angleMode === constants.RADIANS) {
    return Math.sin(angle);
  } else {
    return Math.sin(this.radians(angle));
  }
};

/**
 * Calculates the tangent of an angle. This function takes into account
 * the current angleMode. Values are returned in the range -1 to 1.
 *
 * @method tan
 * @param  {Number} angle the angle
 * @return {Number}       the tangent of the angle
 *
 * @example
 * <div>
 * <code>
 *   var a = 0.0;
 *   var inc = TWO_PI/50.0;
 *   for (var i = 0; i < 100; i = i+2) {
 *     line(i, 50, i, 50+tan(a)*2.0);
 *     a = a + inc;
 *   }
 * </code>
 * </div>
 *
 */
p5.prototype.tan = function(angle) {
  if (this._angleMode === constants.RADIANS) {
    return Math.tan(angle);
  } else {
    return Math.tan(this.radians(angle));
  }
};

/**
 * Converts a radian measurement to its corresponding value in degrees.
 * Radians and degrees are two ways of measuring the same thing. There are
 * 360 degrees in a circle and 2*PI radians in a circle. For example,
 * 90° = PI/2 = 1.5707964.
 *
 * @method degrees
 * @param  {Number} radians the radians value to convert to degrees
 * @return {Number}         the converted angle
 *
 *
 * @example
 * <div class= “norender">
 * <code>
 * var rad = PI/4;
 * var deg = degrees(rad);
 * println(rad + " radians is " + deg + " degrees");
 * // Prints: 45 degrees is 0.7853981633974483 radians
 * </code>
 * </div>
 *
 */
p5.prototype.degrees = function(angle) {
  return polarGeometry.radiansToDegrees(angle);
};

/**
 * Converts a degree measurement to its corresponding value in radians.
 * Radians and degrees are two ways of measuring the same thing. There are
 * 360 degrees in a circle and 2*PI radians in a circle. For example,
 * 90° = PI/2 = 1.5707964.
 *
 * @method radians
 * @param  {Number} degrees the degree value to convert to radians
 * @return {Number}         the converted angle
 *
 * @example
 * <div class= “norender">
 * <code>
 * var deg = 45.0;
 * var rad = radians(deg);
 * println(deg + " degrees is " + rad + " radians");
 * // Prints: 45 degrees is 0.7853981633974483 radians
 * </code>
 * </div>
 */
p5.prototype.radians = function(angle) {
  return polarGeometry.degreesToRadians(angle);
};

/**
 * Sets the current mode of p5 to given mode. Default mode is RADIANS.
 *
 * @method angleMode
 * @param {Number/Constant} mode either RADIANS or DEGREES
 *
 * @example
 * <div>
 * <code>
 * function draw(){
 *   background(204);
 *   angleMode(DEGREES); // Change the mode to DEGREES
 *   var a = atan2(mouseY-height/2, mouseX-width/2);
 *   translate(width/2, height/2);
 *   push();
 *   rotate(a);
 *   rect(-20, -5, 40, 10); // Larger rectangle is rotating in degrees
 *   pop();
 *   angleMode(RADIANS); // Change the mode to RADIANS
 *   rotate(a); // var a stays the same
 *   rect(-40, -5, 20, 10); // Smaller rectangle is rotating in radians
 * }
 * </code>
 * </div>
 *
 */
p5.prototype.angleMode = function(mode) {
  if (mode === constants.DEGREES || mode === constants.RADIANS) {
    this._angleMode = mode;
  }
};

module.exports = p5;

},{"../core/constants":36,"../core/core":37,"./polargeometry":67}],70:[function(_dereq_,module,exports){
/**
 * @module Typography
 * @submodule Attributes
 * @for p5
 * @requires core
 * @requires constants
 */

'use strict';

var p5 = _dereq_('../core/core');

/**
 * Sets the current alignment for drawing text. Accepts two
 * arguments: horizAlign (LEFT, CENTER, or RIGHT) and
 * vertAlign (TOP, BOTTOM, CENTER, or BASELINE).
 *
 * The horizAlign parameter is in reference to the x value
 * of the text() function, while the vertAlign parameter is
 * in reference to the y value.
 *
 * So if you write textAlign(LEFT), you are aligning the left
 * edge of your text to the x value you give in text(). If you
 * write textAlign(RIGHT, TOP), you are aligning the right edge
 * of your text to the x value and the top of edge of the text
 * to the y value.
 *
 * @method textAlign
 * @param {Number/Constant} horizAlign horizontal alignment, either LEFT,
 *                            CENTER, or RIGHT
 * @param {Number/Constant} vertAlign vertical alignment, either TOP,
 *                            BOTTOM, CENTER, or BASELINE
 * @return {Number}
 * @example
 * <div>
 * <code>
 * textSize(16);
 * textAlign(RIGHT);
 * text("ABCD", 50, 30);
 * textAlign(CENTER);
 * text("EFGH", 50, 50);
 * textAlign(LEFT);
 * text("IJKL", 50, 70);
 * </code>
 * </div>
 */
p5.prototype.textAlign = function(horizAlign, vertAlign) {
  return this._renderer.textAlign.apply(this._renderer, arguments);
};

/**
 * Sets/gets the spacing, in pixels, between lines of text. This
 * setting will be used in all subsequent calls to the text() function.
 *
 * @method textLeading
 * @param {Number} leading the size in pixels for spacing between lines
 * @return {Object|Number}
 * @example
 * <div>
 * <code>
 * // Text to display. The "\n" is a "new line" character
 * lines = "L1\nL2\nL3";
 * textSize(12);
 *
 * textLeading(10);  // Set leading to 10
 * text(lines, 10, 25);
 *
 * textLeading(20);  // Set leading to 20
 * text(lines, 40, 25);
 *
 * textLeading(30);  // Set leading to 30
 * text(lines, 70, 25);
 * </code>
 * </div>
 */
p5.prototype.textLeading = function(theLeading) {
  return this._renderer.textLeading.apply(this._renderer, arguments);
};

/**
 * Sets/gets the current font size. This size will be used in all subsequent
 * calls to the text() function. Font size is measured in pixels.
 *
 * @method textSize
 * @param {Number} theSize the size of the letters in units of pixels
 * @return {Object|Number}
 * @example
 * <div>
 * <code>
 * textSize(12);
 * text("Font Size 12", 10, 30);
 * textSize(14);
 * text("Font Size 14", 10, 60);
 * textSize(16);
 * text("Font Size 16", 10, 90);
 * </code>
 * </div>
 */
p5.prototype.textSize = function(theSize) {
  return this._renderer.textSize.apply(this._renderer, arguments);
};

/**
 * Sets/gets the style of the text for system fonts to NORMAL, ITALIC, or BOLD.
 * Note: this may be is overridden by CSS styling. For non-system fonts
 * (opentype, truetype, etc.) please load styled fonts instead.
 *
 * @method textStyle
 * @param {Number/Constant} theStyle styling for text, either NORMAL,
 *                            ITALIC, or BOLD
 * @return {Object|String}
 * @example
 * <div>
 * <code>
 * strokeWeight(0);
 * textSize(12);
 * textStyle(NORMAL);
 * text("Font Style Normal", 10, 30);
 * textStyle(ITALIC);
 * text("Font Style Italic", 10, 60);
 * textStyle(BOLD);
 * text("Font Style Bold", 10, 90);
 * </code>
 * </div>
 */
p5.prototype.textStyle = function(theStyle) {
  return this._renderer.textStyle.apply(this._renderer, arguments);
};

/**
 * Calculates and returns the width of any character or text string.
 *
 * @method textWidth
 * @param {String} theText the String of characters to measure
 * @return {Number}
 * @example
 * <div>
 * <code>
 * textSize(28);
 *
 * var aChar = 'P';
 * var cWidth = textWidth(aChar);
 * text(aChar, 0, 40);
 * line(cWidth, 0, cWidth, 50);
 *
 * var aString = "p5.js";
 * var sWidth = textWidth(aString);
 * text(aString, 0, 85);
 * line(sWidth, 50, sWidth, 100);
 * </code>
 * </div>
 */
p5.prototype.textWidth = function(theText) {
  return this._renderer.textWidth.apply(this._renderer, arguments);
};

/**
 * Returns the ascent of the current font at its current size. The ascent
 * represents the distance, in pixels, of the tallest character above
 * the baseline.
 *
 * @return {Number}
 * @example
 * <div>
 * <code>
 * var base = height * 0.75;
 * var scalar = 0.8; // Different for each font
 *
 * textSize(32);  // Set initial text size
 * var asc = textAscent() * scalar;  // Calc ascent
 * line(0, base - asc, width, base - asc);
 * text("dp", 0, base);  // Draw text on baseline
 *
 * textSize(64);  // Increase text size
 * asc = textAscent() * scalar;  // Recalc ascent
 * line(40, base - asc, width, base - asc);
 * text("dp", 40, base);  // Draw text on baseline
 * </code>
 * </div>
 */
p5.prototype.textAscent = function() {
  return this._renderer.textAscent();
};

/**
 * Returns the descent of the current font at its current size. The descent
 * represents the distance, in pixels, of the character with the longest
 * descender below the baseline.
 *
 * @return {Number}
 * @example
 * <div>
 * <code>
 * var base = height * 0.75;
 * var scalar = 0.8; // Different for each font
 *
 * textSize(32);  // Set initial text size
 * var desc = textDescent() * scalar;  // Calc ascent
 * line(0, base+desc, width, base+desc);
 * text("dp", 0, base);  // Draw text on baseline
 *
 * textSize(64);  // Increase text size
 * desc = textDescent() * scalar;  // Recalc ascent
 * line(40, base + desc, width, base + desc);
 * text("dp", 40, base);  // Draw text on baseline
 * </code>
 * </div>
 */
p5.prototype.textDescent = function() {
  return this._renderer.textDescent();
};

/**
 * Helper function to measure ascent and descent.
 */
p5.prototype._updateTextMetrics = function() {
  return this._renderer._updateTextMetrics();
};

module.exports = p5;

},{"../core/core":37}],71:[function(_dereq_,module,exports){
/**
 * @module Typography
 * @submodule Loading & Displaying
 * @for p5
 * @requires core
 */

'use strict';

var p5 = _dereq_('../core/core');
var constants = _dereq_('../core/constants');

_dereq_('../core/error_helpers');


/**
 * Draws text to the screen. Displays the information specified in the first
 * parameter on the screen in the position specified by the additional
 * parameters. A default font will be used unless a font is set with the
 * textFont() function and a default size will be used unless a font is set
 * with textSize(). Change the color of the text with the fill() function.
 * Change the outline of the text with the stroke() and strokeWeight()
 * functions.
 * <br><br>
 * The text displays in relation to the textAlign() function, which gives the
 * option to draw to the left, right, and center of the coordinates.
 * <br><br>
 * The x2 and y2 parameters define a rectangular area to display within and
 * may only be used with string data. When these parameters are specified,
 * they are interpreted based on the current rectMode() setting. Text that
 * does not fit completely within the rectangle specified will not be drawn
 * to the screen.
 *
 * @method text
 * @param {String} str the alphanumeric symbols to be displayed
 * @param {Number} x   x-coordinate of text
 * @param {Number} y   y-coordinate of text
 * @param {Number} x2  by default, the width of the text box,
 *                     see rectMode() for more info
 * @param {Number} y2  by default, the height of the text box,
 *                     see rectMode() for more info
 * @return {Object} this
 * @example
 * <div>
 * <code>
 * textSize(32);
 * text("word", 10, 30);
 * fill(0, 102, 153);
 * text("word", 10, 60);
 * fill(0, 102, 153, 51);
 * text("word", 10, 90);
 * </code>
 * </div>
 * <div>
 * <code>
 * s = "The quick brown fox jumped over the lazy dog.";
 * fill(50);
 * text(s, 10, 10, 70, 80); // Text wraps within text box
 * </code>
 * </div>
 */
p5.prototype.text = function(str, x, y, maxWidth, maxHeight) {
  var args = new Array(arguments.length);
  for (var i = 0; i < args.length; ++i) {
    args[i] = arguments[i];
  }
  this._validateParameters(
    'text',
    args,
    [
      ['*', 'Number', 'Number'],
      ['*', 'Number', 'Number', 'Number', 'Number']
    ]
  );

  return (!(this._renderer._doFill || this._renderer._doStroke)) ? this :
    this._renderer.text.apply(this._renderer, arguments);
};

/**
 * Sets the current font that will be drawn with the text() function.
 *
 * @method textFont
 * @param {Object|String} f a font loaded via loadFont(), or a String
 *  representing a browser-based default font.
 * @return {Object} this
 * @example
 * <div>
 * <code>
 * fill(0);
 * textSize(12);
 * textFont("Georgia");
 * text("Georgia", 12, 30);
 * textFont("Helvetica");
 * text("Helvetica", 12, 60);
 * </code>
 * </div>
 * <div>
 * <code>
 * var fontRegular, fontItalic, fontBold;
 * function preload() {
 *    fontRegular = loadFont("assets/Regular.otf");
 *    fontItalic = loadFont("assets/Italic.ttf");
 *    fontBold = loadFont("assets/Bold.ttf");
 * }
 * function setup() {
 *    background(210);
 *    fill(0).strokeWeight(0).textSize(10);
 *    textFont(fontRegular);
 *    text("Font Style Normal", 10, 30);
 *    textFont(fontItalic);
 *    text("Font Style Italic", 10, 50);
 *    textFont(fontBold);
 *    text("Font Style Bold", 10, 70);
 * }
 * </code>
 * </div>
 */
p5.prototype.textFont = function(theFont, theSize) {

  if (arguments.length) {

    if (!theFont) {

      throw Error('null font passed to textFont');
    }

    this._renderer._setProperty('_textFont', theFont);

    if (theSize) {

      this._renderer._setProperty('_textSize', theSize);
      this._renderer._setProperty('_textLeading',
        theSize * constants._DEFAULT_LEADMULT);
    }

    return this._renderer._applyTextProperties();
  }

  return this;
};

module.exports = p5;

},{"../core/constants":36,"../core/core":37,"../core/error_helpers":40}],72:[function(_dereq_,module,exports){
/**
 * This module defines the p5.Font class and functions for
 * drawing text to the display canvas.
 * @module Typography
 * @submodule Font
 * @requires core
 * @requires constants
 */

'use strict';

var p5 = _dereq_('../core/core');
var constants = _dereq_('../core/constants');

/*
 * TODO:
 *
 * API:
 * -- textBounds()
 * -- getPath()
 * -- getPoints()
 *
 * ===========================================
 * -- PFont functions:
 *    PFont.list()
 *
 * -- kerning
 * -- alignment: justified?
 * -- integrate p5.dom? (later)
 */

/**
 * Base class for font handling
 * @class p5.Font
 * @constructor
 * @param {Object} [pInst] pointer to p5 instance
 */
p5.Font = function(p) {

  this.parent = p;

  this.cache = {};

  /**
   * Underlying opentype font implementation
   * @property font
   */
  this.font = undefined;
};

p5.Font.prototype.list = function() {

  // TODO
  throw 'not yet implemented';
};

/**
 * Returns a tight bounding box for the given text string using this
 * font (currently only supports single lines)
 *
 * @method textBounds
 * @param  {String} line     a line of text
 * @param  {Number} x        x-position
 * @param  {Number} y        y-position
 * @param  {Number} fontSize font size to use (optional)
 * @param  {Object} options opentype options (optional)
 *
 * @return {Object}          a rectangle object with properties: x, y, w, h
 *
 * @example
 * <div>
 * <code>
 * var font;
 * var textString = 'Lorem ipsum dolor sit amet.';
 * function preload() {
 *    font = loadFont('./assets/Regular.otf');
 * };
 * function setup() {
 *    background(210);
 *
 *    var bbox = font.textBounds(textString, 10, 30, 12);
 *    fill(255);
 *    stroke(0);
 *    rect(bbox.x, bbox.y, bbox.w, bbox.h);
 *    fill(0);
 *    noStroke();
 *
 *    textFont(font);
 *    textSize(12);
 *    text(textString, 10, 30);
 * };
 * </code>
 * </div>
 */
p5.Font.prototype.textBounds = function(str, x, y, fontSize, options) {

  x = x !== undefined ? x : 0;
  y = y !== undefined ? y : 0;
  fontSize = fontSize || this.parent._renderer._textSize;

  // Check cache for existing bounds. Take into consideration the text alignment
  // settings. Default alignment should match opentype's origin: left-aligned &
  // alphabetic baseline.
  var p = (options && options.renderer && options.renderer._pInst) ||
    this.parent,
    ctx = p._renderer.drawingContext,
    alignment = ctx.textAlign || constants.LEFT,
    baseline = ctx.textBaseline || constants.BASELINE;
  var result = this.cache[cacheKey('textBounds', str, x, y, fontSize, alignment,
    baseline)];

  if (!result) {

    var xCoords = [], yCoords = [], self = this,
      scale = this._scale(fontSize), minX, minY, maxX, maxY;

    this.font.forEachGlyph(str, x, y, fontSize, options,
      function(glyph, gX, gY, gFontSize) {

        xCoords.push(gX);
        yCoords.push(gY);

        var gm = glyph.getMetrics();

        if (glyph.name !== 'space') {

          xCoords.push(gX + (gm.xMax * scale));
          yCoords.push(gY + (-gm.yMin * scale));
          yCoords.push(gY + (-gm.yMax * scale));

        } else { // NOTE: deals with broken metrics for spaces in opentype.js

          xCoords.push(gX + self.font.charToGlyph(' ').advanceWidth *
            self._scale(fontSize));
        }
      });

    // fix to #1409 (not sure why these max() functions were here)
    /*minX = Math.max(0, Math.min.apply(null, xCoords));
    minY = Math.max(0, Math.min.apply(null, yCoords));
    maxX = Math.max(0, Math.max.apply(null, xCoords));
    maxY = Math.max(0, Math.max.apply(null, yCoords));*/
    minX = Math.min.apply(null, xCoords);
    minY = Math.min.apply(null, yCoords);
    maxX = Math.max.apply(null, xCoords);
    maxY = Math.max.apply(null, yCoords);

    result = {
      x: minX,
      y: minY,
      h: maxY - minY,
      w: maxX - minX,
      advance: minX - x
    };

    // Bounds are now calculated, so shift the x & y to match alignment settings
    var textWidth = result.w + result.advance;
    var pos = this._handleAlignment(p, ctx, str, result.x, result.y, textWidth);
    result.x = pos.x;
    result.y = pos.y;

    this.cache[cacheKey('textBounds', str, x, y, fontSize, alignment,
      baseline)] = result;
  }
  //else console.log('cache-hit');

  return result;
};


/**
 * Computes an array of points following the path for specified text
 *
 * @param  {String} txt     a line of text
 * @param  {Number} x        x-position
 * @param  {Number} y        y-position
 * @param  {Number} fontSize font size to use (optional)
 * @param  {Object} options  an (optional) object that can contain:
 *
 * <br>sampleFactor - the ratio of path-length to number of samples
 * (default=.25); higher values yield more points and are therefore
 * more precise
 *
 * <br>simplifyThreshold - if set to a non-zero value, collinear points will be
 * be removed from the polygon; the value represents the threshold angle to use
 * when determining whether two edges are collinear
 *
 * @return {Array}  an array of points, each with x, y, alpha (the path angle)
 */
p5.Font.prototype.textToPoints = function(txt, x, y, fontSize, options) {

  var xoff = 0, result = [], glyphs = this._getGlyphs(txt);

  fontSize = fontSize || this.parent._renderer._textSize;

  for (var i = 0; i < glyphs.length; i++) {

    var gpath = glyphs[i].getPath(x, y, fontSize),
      paths = splitPaths(gpath.commands);

    for (var j = 0; j < paths.length; j++) {

      var pts = pathToPoints(paths[j], options);

      for (var k = 0; k < pts.length; k++) {
        pts[k].x += xoff;
        result.push(pts[k]);
      }
    }

    xoff += glyphs[i].advanceWidth * this._scale(fontSize);
  }

  return result;
};

// ----------------------------- End API ------------------------------

/**
 * Returns the set of opentype glyphs for the supplied string.
 *
 * Note that there is not a strict one-to-one mapping between characters
 * and glyphs, so the list of returned glyphs can be larger or smaller
 *  than the length of the given string.
 *
 * @param  {String} str the string to be converted
 * @return {array}     the opentype glyphs
 */
p5.Font.prototype._getGlyphs = function(str) {

  return this.font.stringToGlyphs(str);
};

/**
 * Returns an opentype path for the supplied string and position.
 *
 * @param  {String} line     a line of text
 * @param  {Number} x        x-position
 * @param  {Number} y        y-position
 * @param  {Object} options opentype options (optional)
 * @return {Object}     the opentype path
 */
p5.Font.prototype._getPath = function(line, x, y, options) {

  var p = (options && options.renderer && options.renderer._pInst) ||
    this.parent,
    ctx = p._renderer.drawingContext,
    pos = this._handleAlignment(p, ctx, line, x, y);

  return this.font.getPath(line, pos.x, pos.y, p._renderer._textSize, options);
};

/*
 * Creates an SVG-formatted path-data string
 * (See http://www.w3.org/TR/SVG/paths.html#PathData)
 * from the given opentype path or string/position
 *
 * @param  {Object} path    an opentype path, OR the following:
 *
 * @param  {String} line     a line of text
 * @param  {Number} x        x-position
 * @param  {Number} y        y-position
 * @param  {Object} options opentype options (optional), set options.decimals
 * to set the decimal precision of the path-data
 *
 * @return {Object}     this p5.Font object
 */
p5.Font.prototype._getPathData = function(line, x, y, options) {

  var decimals = 3;

  // create path from string/position
  if (typeof line === 'string' && arguments.length > 2) {

    line = this._getPath(line, x, y, options);
  }
  // handle options specified in 2nd arg
  else if (typeof x === 'object') {

    options = x;
  }

  // handle svg arguments
  if (options && typeof options.decimals === 'number') {

    decimals = options.decimals;
  }

  return line.toPathData(decimals);
};

/*
 * Creates an SVG <path> element, as a string,
 * from the given opentype path or string/position
 *
 * @param  {Object} path    an opentype path, OR the following:
 *
 * @param  {String} line     a line of text
 * @param  {Number} x        x-position
 * @param  {Number} y        y-position
 * @param  {Object} options opentype options (optional), set options.decimals
 * to set the decimal precision of the path-data in the <path> element,
 *  options.fill to set the fill color for the <path> element,
 *  options.stroke to set the stroke color for the <path> element,
 *  options.strokeWidth to set the strokeWidth for the <path> element.
 *
 * @return {Object}     this p5.Font object
 */
p5.Font.prototype._getSVG = function(line, x, y, options) {

  var decimals = 3;

  // create path from string/position
  if (typeof line === 'string' && arguments.length > 2) {

    line = this._getPath(line, x, y, options);
  }
  // handle options specified in 2nd arg
  else if (typeof x === 'object') {

    options = x;
  }

  // handle svg arguments
  if (options) {
    if (typeof options.decimals === 'number') {
      decimals = options.decimals;
    }
    if (typeof options.strokeWidth === 'number') {
      line.strokeWidth = options.strokeWidth;
    }
    if (typeof options.fill !== 'undefined') {
      line.fill = options.fill;
    }
    if (typeof options.stroke !== 'undefined') {
      line.stroke = options.stroke;
    }
  }

  return line.toSVG(decimals);
};

/*
 * Renders an opentype path or string/position
 * to the current graphics context
 *
 * @param  {Object} path    an opentype path, OR the following:
 *
 * @param  {String} line     a line of text
 * @param  {Number} x        x-position
 * @param  {Number} y        y-position
 * @param  {Object} options opentype options (optional)
 *
 * @return {Object}     this p5.Font object
 */
p5.Font.prototype._renderPath = function(line, x, y, options) {

  var pdata, pg = (options && options.renderer) || this.parent._renderer,
    ctx = pg.drawingContext;

  if (typeof line === 'object' && line.commands) {

    pdata = line.commands;
  } else {

    //pos = handleAlignment(p, ctx, line, x, y);
    pdata = this._getPath(line, x, y, options).commands;
  }

  ctx.beginPath();
  for (var i = 0; i < pdata.length; i += 1) {

    var cmd = pdata[i];
    if (cmd.type === 'M') {
      ctx.moveTo(cmd.x, cmd.y);
    } else if (cmd.type === 'L') {
      ctx.lineTo(cmd.x, cmd.y);
    } else if (cmd.type === 'C') {
      ctx.bezierCurveTo(cmd.x1, cmd.y1, cmd.x2, cmd.y2, cmd.x, cmd.y);
    } else if (cmd.type === 'Q') {
      ctx.quadraticCurveTo(cmd.x1, cmd.y1, cmd.x, cmd.y);
    } else if (cmd.type === 'Z') {
      ctx.closePath();
    }
  }

  // only draw stroke if manually set by user
  if (pg._doStroke && pg._strokeSet) {

    ctx.stroke();
  }

  if (pg._doFill) {

    // if fill hasn't been set by user, use default-text-fill
    ctx.fillStyle = pg._fillSet ? ctx.fillStyle : constants._DEFAULT_TEXT_FILL;
    ctx.fill();
  }

  return this;
};

p5.Font.prototype._textWidth = function(str, fontSize) {

  if (str === ' ') { // special case for now

    return this.font.charToGlyph(' ').advanceWidth * this._scale(fontSize);
  }

  var bounds = this.textBounds(str, 0, 0, fontSize);
  return bounds.w + bounds.advance;
};

p5.Font.prototype._textAscent = function(fontSize) {

  return this.font.ascender * this._scale(fontSize);
};

p5.Font.prototype._textDescent = function(fontSize) {

  return -this.font.descender * this._scale(fontSize);
};

p5.Font.prototype._scale = function(fontSize) {

  return (1 / this.font.unitsPerEm) * (fontSize ||
    this.parent._renderer._textSize);
};

p5.Font.prototype._handleAlignment = function(p, ctx, line, x, y, textWidth) {
  var fontSize = p._renderer._textSize,
    textAscent = this._textAscent(fontSize),
    textDescent = this._textDescent(fontSize);

  textWidth = textWidth !== undefined ? textWidth :
    this._textWidth(line, fontSize);

  if (ctx.textAlign === constants.CENTER) {
    x -= textWidth / 2;
  } else if (ctx.textAlign === constants.RIGHT) {
    x -= textWidth;
  }

  if (ctx.textBaseline === constants.TOP) {
    y += textAscent;
  } else if (ctx.textBaseline === constants._CTX_MIDDLE) {
    y += textAscent / 2;
  } else if (ctx.textBaseline === constants.BOTTOM) {
    y -= textDescent;
  }

  return { x: x, y: y };
};

// path-utils

function pathToPoints(cmds, options) {

  var opts = parseOpts(options, {
    sampleFactor: 0.1,
    simplifyThreshold: 0,
  });

  var len = pointAtLength(cmds,0,1), // total-length
    t = len / (len * opts.sampleFactor),
    pts = [];

  for (var i = 0; i < len; i += t) {
    pts.push(pointAtLength(cmds, i));
  }

  if (opts.simplifyThreshold) {
    /*var count = */simplify(pts, opts.simplifyThreshold);
    //console.log('Simplify: removed ' + count + ' pts');
  }

  return pts;
}

function simplify(pts, angle) {

  angle = (typeof angle === 'undefined') ? 0 : angle;

  var num = 0;
  for (var i = pts.length - 1; pts.length > 3 && i >= 0; --i) {

    if (collinear(at(pts, i - 1), at(pts, i), at(pts, i + 1), angle)) {

      // Remove the middle point
      pts.splice(i % pts.length, 1);
      num++;
    }
  }
  return num;
}

function splitPaths(cmds) {

  var paths = [], current;
  for (var i = 0; i < cmds.length; i++) {
    if (cmds[i].type === 'M') {
      if (current) {
        paths.push(current);
      }
      current = [];
    }
    current.push(cmdToArr(cmds[i]));
  }
  paths.push(current);

  return paths;
}

function cmdToArr(cmd) {

  var arr = [ cmd.type ];
  if (cmd.type === 'M' || cmd.type === 'L') { // moveto or lineto
    arr.push(cmd.x, cmd.y);
  } else if (cmd.type === 'C') {
    arr.push(cmd.x1, cmd.y1, cmd.x2, cmd.y2, cmd.x, cmd.y);
  } else if (cmd.type === 'Q') {
    arr.push(cmd.x1, cmd.y1, cmd.x, cmd.y);
  }
  // else if (cmd.type === 'Z') { /* no-op */ }
  return arr;
}

function parseOpts(options, defaults) {

  if (typeof options !== 'object') {
    options = defaults;
  }
  else {
    for (var key in defaults) {
      if (typeof options[key] === 'undefined') {
        options[key] = defaults[key];
      }
    }
  }
  return options;
}

//////////////////////// Helpers ////////////////////////////

function at(v, i) {
  var s = v.length;
  return v[i < 0 ? i % s + s : i % s];
}

function collinear(a, b, c, thresholdAngle) {

  if (!thresholdAngle) {
    return areaTriangle(a, b, c) === 0;
  }

  if (typeof collinear.tmpPoint1 === 'undefined') {
    collinear.tmpPoint1 = [];
    collinear.tmpPoint2 = [];
  }

  var ab = collinear.tmpPoint1, bc = collinear.tmpPoint2;
  ab.x = b.x - a.x;
  ab.y = b.y - a.y;
  bc.x = c.x - b.x;
  bc.y = c.y - b.y;

  var dot = ab.x * bc.x + ab.y * bc.y,
    magA = Math.sqrt(ab.x * ab.x + ab.y * ab.y),
    magB = Math.sqrt(bc.x * bc.x + bc.y * bc.y),
    angle = Math.acos(dot / (magA * magB));

  return angle < thresholdAngle;
}

function areaTriangle(a, b, c) {
  return (((b[0] - a[0]) * (c[1] - a[1])) - ((c[0] - a[0]) * (b[1] - a[1])));
}

// Portions of below code copyright 2008 Dmitry Baranovskiy (via MIT license)

function findDotsAtSegment(p1x, p1y, c1x, c1y, c2x, c2y, p2x, p2y, t) {

  var t1 = 1 - t, t13 = Math.pow(t1, 3), t12 = Math.pow(t1, 2), t2 = t * t,
    t3 = t2 * t, x = t13 * p1x + t12 * 3 * t * c1x + t1 * 3 * t * t * c2x +
    t3 * p2x, y = t13 * p1y + t12 * 3 * t * c1y + t1 * 3 * t * t * c2y +
    t3 * p2y, mx = p1x + 2 * t * (c1x - p1x) + t2 * (c2x - 2 * c1x + p1x),
    my = p1y + 2 * t * (c1y - p1y) + t2 * (c2y - 2 * c1y + p1y),
    nx = c1x + 2 * t * (c2x - c1x) + t2 * (p2x - 2 * c2x + c1x),
    ny = c1y + 2 * t * (c2y - c1y) + t2 * (p2y - 2 * c2y + c1y),
    ax = t1 * p1x + t * c1x, ay = t1 * p1y + t * c1y,
    cx = t1 * c2x + t * p2x, cy = t1 * c2y + t * p2y,
    alpha = (90 - Math.atan2(mx - nx, my - ny) * 180 / Math.PI);

  if (mx > nx || my < ny) { alpha += 180; }

  return { x: x, y: y, m: { x: mx, y: my }, n: { x: nx, y: ny },
    start: { x: ax, y: ay }, end: { x: cx, y: cy }, alpha: alpha
  };
}

function getPointAtSegmentLength(p1x,p1y,c1x,c1y,c2x,c2y,p2x,p2y,length) {
  return (length == null) ? bezlen(p1x, p1y, c1x, c1y, c2x, c2y, p2x, p2y) :
    findDotsAtSegment(p1x, p1y, c1x, c1y, c2x, c2y, p2x, p2y,
      getTatLen(p1x, p1y, c1x, c1y, c2x, c2y, p2x, p2y, length));
}

function pointAtLength(path, length, istotal) {
  path = path2curve(path);
  var x, y, p, l, sp = '', subpaths = {}, point, len = 0;
  for (var i = 0, ii = path.length; i < ii; i++) {
    p = path[i];
    if (p[0] === 'M') {
      x = +p[1];
      y = +p[2];
    } else {
      l = getPointAtSegmentLength(x, y, p[1], p[2], p[3], p[4], p[5], p[6]);
      if (len + l > length) {
        if (!istotal) {
          point = getPointAtSegmentLength(x, y, p[1], p[2], p[3], p[4], p[5],
            p[6], length - len);
          return { x: point.x, y: point.y, alpha: point.alpha };
        }
      }
      len += l;
      x = +p[5];
      y = +p[6];
    }
    sp += p.shift() + p;
  }
  subpaths.end = sp;

  point = istotal ? len : findDotsAtSegment
    (x, y, p[0], p[1], p[2], p[3], p[4], p[5], 1);

  if (point.alpha) {
    point = { x: point.x, y: point.y, alpha: point.alpha };
  }

  return point;
}

function pathToAbsolute(pathArray) {

  var res = [], x = 0, y = 0, mx = 0, my = 0, start = 0;
  if (pathArray[0][0] === 'M') {
    x = +pathArray[0][1];
    y = +pathArray[0][2];
    mx = x;
    my = y;
    start++;
    res[0] = ['M', x, y];
  }

  var dots,crz = pathArray.length===3 && pathArray[0][0]==='M' &&
    pathArray[1][0].toUpperCase()==='R' && pathArray[2][0].toUpperCase()==='Z';

  for (var r, pa, i = start, ii = pathArray.length; i < ii; i++) {
    res.push(r = []);
    pa = pathArray[i];
    if (pa[0] !== String.prototype.toUpperCase.call(pa[0])) {
      r[0] = String.prototype.toUpperCase.call(pa[0]);
      switch (r[0]) {
        case 'A':
          r[1] = pa[1];
          r[2] = pa[2];
          r[3] = pa[3];
          r[4] = pa[4];
          r[5] = pa[5];
          r[6] = +(pa[6] + x);
          r[7] = +(pa[7] + y);
          break;
        case 'V':
          r[1] = +pa[1] + y;
          break;
        case 'H':
          r[1] = +pa[1] + x;
          break;
        case 'R':
          dots = [x, y].concat(pa.slice(1));
          for (var j = 2, jj = dots.length; j < jj; j++) {
            dots[j] = +dots[j] + x;
            dots[++j] = +dots[j] + y;
          }
          res.pop();
          res = res.concat(catmullRom2bezier(dots, crz));
          break;
        case 'M':
          mx = +pa[1] + x;
          my = +pa[2] + y;
          break;
        default:
          for (j = 1, jj = pa.length; j < jj; j++) {
            r[j] = +pa[j] + ((j % 2) ? x : y);
          }
      }
    } else if (pa[0] === 'R') {
      dots = [x, y].concat(pa.slice(1));
      res.pop();
      res = res.concat(catmullRom2bezier(dots, crz));
      r = ['R'].concat(pa.slice(-2));
    } else {
      for (var k = 0, kk = pa.length; k < kk; k++) {
        r[k] = pa[k];
      }
    }
    switch (r[0]) {
      case 'Z':
        x = mx;
        y = my;
        break;
      case 'H':
        x = r[1];
        break;
      case 'V':
        y = r[1];
        break;
      case 'M':
        mx = r[r.length - 2];
        my = r[r.length - 1];
        break;
      default:
        x = r[r.length - 2];
        y = r[r.length - 1];
    }
  }
  return res;
}

function path2curve(path, path2) {

  var p = pathToAbsolute(path), p2 = path2 && pathToAbsolute(path2),
    attrs = { x: 0, y: 0, bx: 0, by: 0, X: 0, Y: 0, qx: null, qy: null },
    attrs2 = { x: 0, y: 0, bx: 0, by: 0, X: 0, Y: 0, qx: null, qy: null },

    processPath = function(path, d, pcom) {
      var nx, ny, tq = { T: 1, Q: 1 };
      if (!path) { return ['C', d.x, d.y, d.x, d.y, d.x, d.y]; }
      if (!(path[0] in tq)) { d.qx = d.qy = null; }
      switch (path[0]) {
        case 'M':
          d.X = path[1];
          d.Y = path[2];
          break;
        case 'A':
          path = ['C'].concat(a2c.apply(0, [d.x, d.y].concat(path.slice(1))));
          break;
        case 'S':
          if (pcom === 'C' || pcom === 'S') {
            nx = d.x * 2 - d.bx;
            ny = d.y * 2 - d.by;
          } else {
            nx = d.x;
            ny = d.y;
          }
          path = ['C', nx, ny].concat(path.slice(1));
          break;
        case 'T':
          if (pcom === 'Q' || pcom === 'T') {
            d.qx = d.x * 2 - d.qx;
            d.qy = d.y * 2 - d.qy;
          } else {
            d.qx = d.x;
            d.qy = d.y;
          }
          path = ['C'].concat(q2c(d.x, d.y, d.qx, d.qy, path[1], path[2]));
          break;
        case 'Q':
          d.qx = path[1];
          d.qy = path[2];
          path = ['C'].concat(q2c(d.x,d.y,path[1],path[2],path[3],path[4]));
          break;
        case 'L':
          path = ['C'].concat(l2c(d.x, d.y, path[1], path[2]));
          break;
        case 'H':
          path = ['C'].concat(l2c(d.x, d.y, path[1], d.y));
          break;
        case 'V':
          path = ['C'].concat(l2c(d.x, d.y, d.x, path[1]));
          break;
        case 'Z':
          path = ['C'].concat(l2c(d.x, d.y, d.X, d.Y));
          break;
      }
      return path;
    },

    fixArc = function(pp, i) {
      if (pp[i].length > 7) {
        pp[i].shift();
        var pi = pp[i];
        while (pi.length) {
          pcoms1[i] = 'A';
          if (p2) { pcoms2[i] = 'A'; }
          pp.splice(i++, 0, ['C'].concat(pi.splice(0, 6)));
        }
        pp.splice(i, 1);
        ii = Math.max(p.length, p2 && p2.length || 0);
      }
    },

    fixM = function(path1, path2, a1, a2, i) {
      if (path1 && path2 && path1[i][0] === 'M' && path2[i][0] !== 'M') {
        path2.splice(i, 0, ['M', a2.x, a2.y]);
        a1.bx = 0;
        a1.by = 0;
        a1.x = path1[i][1];
        a1.y = path1[i][2];
        ii = Math.max(p.length, p2 && p2.length || 0);
      }
    },

    pcoms1 = [], // path commands of original path p
    pcoms2 = [], // path commands of original path p2
    pfirst = '', // temporary holder for original path command
    pcom = ''; // holder for previous path command of original path

  for (var i = 0, ii = Math.max(p.length, p2 && p2.length || 0); i < ii; i++) {
    if (p[i]) { pfirst = p[i][0]; } // save current path command

    if (pfirst !== 'C') {
      pcoms1[i] = pfirst; // Save current path command
      if (i) { pcom = pcoms1[i - 1]; } // Get previous path command pcom
    }
    p[i] = processPath(p[i], attrs, pcom);

    if (pcoms1[i] !== 'A' && pfirst === 'C') { pcoms1[i] = 'C'; }

    fixArc(p, i); // fixArc adds also the right amount of A:s to pcoms1

    if (p2) { // the same procedures is done to p2
      if (p2[i]) { pfirst = p2[i][0]; }
      if (pfirst !== 'C') {
        pcoms2[i] = pfirst;
        if (i) { pcom = pcoms2[i - 1]; }
      }
      p2[i] = processPath(p2[i], attrs2, pcom);

      if (pcoms2[i] !== 'A' && pfirst === 'C') { pcoms2[i] = 'C'; }

      fixArc(p2, i);
    }
    fixM(p, p2, attrs, attrs2, i);
    fixM(p2, p, attrs2, attrs, i);
    var seg = p[i], seg2 = p2 && p2[i], seglen = seg.length,
      seg2len = p2 && seg2.length;
    attrs.x = seg[seglen - 2];
    attrs.y = seg[seglen - 1];
    attrs.bx = parseFloat(seg[seglen - 4]) || attrs.x;
    attrs.by = parseFloat(seg[seglen - 3]) || attrs.y;
    attrs2.bx = p2 && (parseFloat(seg2[seg2len - 4]) || attrs2.x);
    attrs2.by = p2 && (parseFloat(seg2[seg2len - 3]) || attrs2.y);
    attrs2.x = p2 && seg2[seg2len - 2];
    attrs2.y = p2 && seg2[seg2len - 1];
  }

  return p2 ? [p, p2] : p;
}

function a2c(x1, y1, rx, ry, angle, lac, sweep_flag, x2, y2, recursive) {
  // for more information of where this Math came from visit:
  // http://www.w3.org/TR/SVG11/implnote.html#ArcImplementationNotes
  var PI = Math.PI, _120 = PI * 120 / 180, f1, f2, cx, cy,
    rad = PI / 180 * (+angle || 0), res = [], xy,
    rotate = function (x, y, rad) {
      var X = x * Math.cos(rad) - y * Math.sin(rad),
        Y = x * Math.sin(rad) + y * Math.cos(rad);
      return { x: X, y: Y };
    };
  if (!recursive) {
    xy = rotate(x1, y1, -rad);
    x1 = xy.x;
    y1 = xy.y;
    xy = rotate(x2, y2, -rad);
    x2 = xy.x;
    y2 = xy.y;
    var x = (x1 - x2) / 2, y = (y1 - y2) / 2,
      h = (x * x) / (rx * rx) + (y * y) / (ry * ry);
    if (h > 1) {
      h = Math.sqrt(h);
      rx = h * rx;
      ry = h * ry;
    }
    var rx2 = rx * rx, ry2 = ry * ry,
      k = (lac === sweep_flag ? -1 : 1) * Math.sqrt(Math.abs
        ((rx2 * ry2 - rx2 * y * y - ry2 * x * x)/(rx2 * y * y + ry2 * x * x)));

    cx = k * rx * y / ry + (x1 + x2) / 2;
    cy = k * -ry * x / rx + (y1 + y2) / 2;
    f1 = Math.asin(((y1 - cy) / ry).toFixed(9));
    f2 = Math.asin(((y2 - cy) / ry).toFixed(9));

    f1 = x1 < cx ? PI - f1 : f1;
    f2 = x2 < cx ? PI - f2 : f2;

    if (f1 < 0) { f1 = PI * 2 + f1; }
    if (f2 < 0) { f2 = PI * 2 + f2; }

    if (sweep_flag && f1 > f2) {
      f1 = f1 - PI * 2;
    }
    if (!sweep_flag && f2 > f1) {
      f2 = f2 - PI * 2;
    }
  } else {
    f1 = recursive[0];
    f2 = recursive[1];
    cx = recursive[2];
    cy = recursive[3];
  }
  var df = f2 - f1;
  if (Math.abs(df) > _120) {
    var f2old = f2, x2old = x2, y2old = y2;
    f2 = f1 + _120 * (sweep_flag && f2 > f1 ? 1 : -1);
    x2 = cx + rx * Math.cos(f2);
    y2 = cy + ry * Math.sin(f2);
    res = a2c(x2, y2, rx, ry, angle, 0, sweep_flag, x2old, y2old,
      [f2, f2old, cx, cy]);
  }
  df = f2 - f1;
  var c1 = Math.cos(f1),
    s1 = Math.sin(f1),
    c2 = Math.cos(f2),
    s2 = Math.sin(f2),
    t = Math.tan(df / 4),
    hx = 4 / 3 * rx * t,
    hy = 4 / 3 * ry * t,
    m1 = [x1, y1],
    m2 = [x1 + hx * s1, y1 - hy * c1],
    m3 = [x2 + hx * s2, y2 - hy * c2],
    m4 = [x2, y2];
  m2[0] = 2 * m1[0] - m2[0];
  m2[1] = 2 * m1[1] - m2[1];
  if (recursive) {
    return [m2, m3, m4].concat(res);
  } else {
    res = [m2, m3, m4].concat(res).join().split(',');
    var newres = [];
    for (var i = 0, ii = res.length; i < ii; i++) {
      newres[i] = i % 2 ? rotate(res[i - 1], res[i], rad).y : rotate(res[i],
        res[i + 1], rad).x;
    }
    return newres;
  }
}

// http://schepers.cc/getting-to-the-point
function catmullRom2bezier(crp, z) {
  var d = [];
  for (var i = 0, iLen = crp.length; iLen - 2 * !z > i; i += 2) {
    var p = [{
      x: +crp[i - 2],
      y: +crp[i - 1]
    }, {
      x: +crp[i],
      y: +crp[i + 1]
    }, {
      x: +crp[i + 2],
      y: +crp[i + 3]
    }, {
      x: +crp[i + 4],
      y: +crp[i + 5]
    }];
    if (z) {
      if (!i) {
        p[0] = {
          x: +crp[iLen - 2],
          y: +crp[iLen - 1]
        };
      } else if (iLen - 4 === i) {
        p[3] = {
          x: +crp[0],
          y: +crp[1]
        };
      } else if (iLen - 2 === i) {
        p[2] = {
          x: +crp[0],
          y: +crp[1]
        };
        p[3] = {
          x: +crp[2],
          y: +crp[3]
        };
      }
    } else {
      if (iLen - 4 === i) {
        p[3] = p[2];
      } else if (!i) {
        p[0] = {
          x: +crp[i],
          y: +crp[i + 1]
        };
      }
    }
    d.push(['C', (-p[0].x + 6 * p[1].x + p[2].x) / 6, (-p[0].y + 6 * p[1].y +
      p[2].y) / 6, (p[1].x + 6 * p[2].x - p[3].x) / 6, (p[1].y + 6 * p[2].y -
      p[3].y) / 6, p[2].x, p[2].y ]);
  }

  return d;
}

function l2c(x1, y1, x2, y2) { return [x1, y1, x2, y2, x2, y2]; }

function q2c(x1, y1, ax, ay, x2, y2) {
  var _13 = 1 / 3, _23 = 2 / 3;
  return [
    _13 * x1 + _23 * ax, _13 * y1 + _23 * ay,
    _13 * x2 + _23 * ax, _13 * y2 + _23 * ay, x2, y2
  ];
}

function bezlen(x1, y1, x2, y2, x3, y3, x4, y4, z) {
  if (z == null) { z = 1; }
  z = z > 1 ? 1 : z < 0 ? 0 : z;
  var z2 = z / 2,
    n = 12, Tvalues = [-0.1252, 0.1252, -0.3678, 0.3678, -0.5873, 0.5873,
       -0.7699, 0.7699, -0.9041, 0.9041, -0.9816, 0.9816],
    sum = 0, Cvalues = [0.2491, 0.2491, 0.2335, 0.2335, 0.2032, 0.2032,
      0.1601, 0.1601, 0.1069, 0.1069, 0.0472, 0.0472 ];
  for (var i = 0; i < n; i++) {
    var ct = z2 * Tvalues[i] + z2,
      xbase = base3(ct, x1, x2, x3, x4),
      ybase = base3(ct, y1, y2, y3, y4),
      comb = xbase * xbase + ybase * ybase;
    sum += Cvalues[i] * Math.sqrt(comb);
  }
  return z2 * sum;
}

function getTatLen(x1, y1, x2, y2, x3, y3, x4, y4, ll) {
  if (ll < 0 || bezlen(x1, y1, x2, y2, x3, y3, x4, y4) < ll) {
    return;
  }
  var t = 1, step = t / 2, t2 = t - step, l, e = 0.01;
  l = bezlen(x1, y1, x2, y2, x3, y3, x4, y4, t2);
  while (Math.abs(l - ll) > e) {
    step /= 2;
    t2 += (l < ll ? 1 : -1) * step;
    l = bezlen(x1, y1, x2, y2, x3, y3, x4, y4, t2);
  }
  return t2;
}

function base3(t, p1, p2, p3, p4) {
  var t1 = -3 * p1 + 9 * p2 - 9 * p3 + 3 * p4,
    t2 = t * t1 + 6 * p1 - 12 * p2 + 6 * p3;
  return t * t2 - 3 * p1 + 3 * p2;
}

function cacheKey() {
  var args = new Array(arguments.length);
  for (var i = 0; i < args.length; ++i) {
    args[i] = arguments[i];
  }
  i = args.length;
  var hash = '';
  while (i--) {
    hash += (args[i] === Object(args[i])) ?
      JSON.stringify(args[i]) : args[i];
  }
  return hash;
}

module.exports = p5.Font;

},{"../core/constants":36,"../core/core":37}],73:[function(_dereq_,module,exports){
/**
 * @module Data
 * @submodule Array Functions
 * @for p5
 * @requires core
 */

'use strict';

var p5 = _dereq_('../core/core');

/**
 * Adds a value to the end of an array. Extends the length of
 * the array by one. Maps to Array.push().
 *
 * @method append
 * @param {Array} array Array to append
 * @param {any} value to be added to the Array
 * @example
 * <div class = "norender"><code>
 * function setup() {
 *
 * var myArray = new Array("Mango", "Apple", "Papaya")
 * print(myArray) // ["Mango", "Apple", "Papaya"]
 *
 * append(myArray, "Peach")
 * print(myArray) // ["Mango", "Apple", "Papaya", "Peach"]
 *
 * }
 * </div></code>
 */
p5.prototype.append = function(array, value) {
  array.push(value);
  return array;
};

/**
 * Copies an array (or part of an array) to another array. The src array is
 * copied to the dst array, beginning at the position specified by
 * srcPosition and into the position specified by dstPosition. The number of
 * elements to copy is determined by length. Note that copying values
 * overwrites existing values in the destination array. To append values
 * instead of overwriting them, use concat().
 * <br><br>
 * The simplified version with only two arguments, arrayCopy(src, dst),
 * copies an entire array to another of the same size. It is equivalent to
 * arrayCopy(src, 0, dst, 0, src.length).
 * <br><br>
 * Using this function is far more efficient for copying array data than
 * iterating through a for() loop and copying each element individually.
 *
 * @method arrayCopy
 * @param {Array}  src           the source Array
 * @param {Number} [srcPosition] starting position in the source Array
 * @param {Array}  dst           the destination Array
 * @param {Number} [dstPosition] starting position in the destination Array
 * @param {Number} [length]      number of Array elements to be copied
 *
 * @example
 *  <div class="norender"><code>
 *  function setup() {
 *
 *    var src = new Array("A", "B", "C");
 *    var dst = new Array( 1 ,  2 ,  3 );
 *    var srcPosition = 1;
 *    var dstPosition = 0;
 *    var length = 2;
 *
 *    print(src); // ["A", "B", "C"]
 *    print(dst); // [ 1 ,  2 ,  3 ]
 *
 *    arrayCopy(src, srcPosition, dst, dstPosition, length);
 *    print(dst); // ["B", "C", 3]
 *
 *    }
 *  </div></code>
 */
p5.prototype.arrayCopy = function(
  src,
  srcPosition,
  dst,
  dstPosition,
  length) {

  // the index to begin splicing from dst array
  var start,
      end;

  if (typeof length !== 'undefined') {

    end = Math.min(length, src.length);
    start = dstPosition;
    src = src.slice(srcPosition, end + srcPosition);

  } else {

    if (typeof dst !== 'undefined') { // src, dst, length
      // rename  so we don't get confused
      end = dst;
      end = Math.min(end, src.length);
    } else { // src, dst
      end = src.length;
    }

    start = 0;
    // rename  so we don't get confused
    dst = srcPosition;
    src = src.slice(0, end);
  }

  // Since we are not returning the array and JavaScript is pass by reference
  // we must modify the actual values of the array
  // instead of reassigning arrays
  Array.prototype.splice.apply(dst, [start, end].concat(src));

};

/**
 * Concatenates two arrays, maps to Array.concat(). Does not modify the
 * input arrays.
 *
 * @method concat
 * @param {Array} a first Array to concatenate
 * @param {Array} b second Array to concatenate
 * @return {Array} concatenated array
 *
 * @example
 * <div class = "norender"><code>
 * function setup() {
 *   var arr1 = new Array("A", "B", "C");
 *   var arr2 = new Array( 1 ,  2 ,  3 );
 *
 *   print(arr1); // ["A","B","C"]
 *   print(arr2); // [1,2,3]
 *
 *   var arr3 = concat(arr1, arr2);
 *
 *   print(arr1); // ["A","B","C"]
 *   print(arr2); // [1,2,3]
 *   print(arr3); // ["A","B","C",1,2,3]
 *
 * }
 * </div></code>
 */
p5.prototype.concat = function(list0, list1) {
  return list0.concat(list1);
};

/**
 * Reverses the order of an array, maps to Array.reverse()
 *
 * @method reverse
 * @param {Array} list Array to reverse
 * @example
 * <div class="norender"><code>
 * function setup() {
 *   var myArray = new Array("A", "B", "C");
 *   print(myArray); // ["A","B","C"]
 *
 *   reverse(myArray);
 *   print(myArray); // ["C","B","A"]
 * }
 * </div></code>
 */
p5.prototype.reverse = function(list) {
  return list.reverse();
};

/**
 * Decreases an array by one element and returns the shortened array,
 * maps to Array.pop().
 *
 * @method shorten
 * @param  {Array} list Array to shorten
 * @return {Array} shortened Array
 * @example
 * <div class = "norender"><code>
 * function setup() {
 *   var myArray = new Array("A", "B", "C");
 *   print(myArray); // ["A","B","C"]
 *
 *   var newArray = shorten(myArray);
 *   print(myArray); // ["A","B","C"]
 *   print(newArray); // ["A","B"]
 * }
 * </div></code>
 */
p5.prototype.shorten = function(list) {
  list.pop();
  return list;
};

/**
 * Randomizes the order of the elements of an array. Implements
 * <a href="http://Bost.Ocks.org/mike/shuffle/" target=_blank>
 * Fisher-Yates Shuffle Algorithm</a>.
 *
 * @method shuffle
 * @param  {Array}   array  Array to shuffle
 * @param  {Boolean} [bool] modify passed array
 * @return {Array}   shuffled Array
 * @example
 * <div><code>
 * function setup() {
 *   var regularArr = ['ABC', 'def', createVector(), TAU, Math.E];
 *   print(regularArr);
 *   shuffle(regularArr, true); // force modifications to passed array
 *   print(regularArr);
 *
 *   // By default shuffle() returns a shuffled cloned array:
 *   var newArr = shuffle(regularArr);
 *   print(regularArr);
 *   print(newArr);
 * }
 * </code></div>
 */
p5.prototype.shuffle = function(arr, bool) {
  var isView = ArrayBuffer && ArrayBuffer.isView && ArrayBuffer.isView(arr);
  arr = bool || isView ? arr : arr.slice();

  var rnd, tmp, idx = arr.length;
  while (idx > 1) {
    rnd = Math.random()*idx | 0;

    tmp = arr[--idx];
    arr[idx] = arr[rnd];
    arr[rnd] = tmp;
  }

  return arr;
};

/**
 * Sorts an array of numbers from smallest to largest, or puts an array of
 * words in alphabetical order. The original array is not modified; a
 * re-ordered array is returned. The count parameter states the number of
 * elements to sort. For example, if there are 12 elements in an array and
 * count is set to 5, only the first 5 elements in the array will be sorted.
 *
 * @method sort
 * @param {Array} list Array to sort
 * @param {Number} [count] number of elements to sort, starting from 0
 *
 * @example
 * <div class = "norender"><code>
 * function setup() {
 *   var words = new Array("banana", "apple", "pear","lime");
 *   print(words); // ["banana", "apple", "pear", "lime"]
 *   var count = 4; // length of array
 *
 *   words = sort(words, count);
 *   print(words); // ["apple", "banana", "lime", "pear"]
 * }
 * </div></code>
 * <div class = "norender"><code>
 * function setup() {
 *   var numbers = new Array(2,6,1,5,14,9,8,12);
 *   print(numbers); // [2,6,1,5,14,9,8,12]
 *   var count = 5; // Less than the length of the array
 *
 *   numbers = sort(numbers, count);
 *   print(numbers); // [1,2,5,6,14,9,8,12]
 * }
 * </div></code>
 */
p5.prototype.sort = function(list, count) {
  var arr = count ? list.slice(0, Math.min(count, list.length)) : list;
  var rest = count ? list.slice(Math.min(count, list.length)) : [];
  if (typeof arr[0] === 'string') {
    arr = arr.sort();
  } else {
    arr = arr.sort(function(a,b){return a-b;});
  }
  return arr.concat(rest);
};

/**
 * Inserts a value or an array of values into an existing array. The first
 * parameter specifies the initial array to be modified, and the second
 * parameter defines the data to be inserted. The third parameter is an index
 * value which specifies the array position from which to insert data.
 * (Remember that array index numbering starts at zero, so the first position
 * is 0, the second position is 1, and so on.)
 *
 * @method splice
 * @param {Array}  list Array to splice into
 * @param {any}    value value to be spliced in
 * @param {Number} position in the array from which to insert data
 *
 * @example
 * <div class = "norender"><code>
 * function setup() {
 *   var myArray = new Array(0,1,2,3,4);
 *   var insArray = new Array("A","B","C");
 *   print(myArray); // [0,1,2,3,4]
 *   print(insArray); // ["A","B","C"]
 *
 *   splice(myArray, insArray, 3);
 *   print(myArray); // [0,1,2,"A","B","C",3,4]
 * }
 * </div></code>
 */
p5.prototype.splice = function(list, value, index) {

  // note that splice returns spliced elements and not an array
  Array.prototype.splice.apply(list, [index, 0].concat(value));

  return list;
};

/**
 * Extracts an array of elements from an existing array. The list parameter
 * defines the array from which the elements will be copied, and the start
 * and count parameters specify which elements to extract. If no count is
 * given, elements will be extracted from the start to the end of the array.
 * When specifying the start, remember that the first array element is 0.
 * This function does not change the source array.
 *
 * @method subset
 * @param  {Array}  list    Array to extract from
 * @param  {Number} start   position to begin
 * @param  {Number} [count] number of values to extract
 * @return {Array}          Array of extracted elements
 *
 * @example
 * <div class = "norender"><code>
 * function setup() {
 *   var myArray = new Array(1,2,3,4,5);
 *   print(myArray); // [1,2,3,4,5]
 *
 *   var sub1 = subset(myArray, 0, 3);
 *   var sub2 = subset(myArray, 2, 2);
 *   print(sub1); // [1,2,3]
 *   print(sub2); // [3,4]
 * }
 * </div></code>
 */
p5.prototype.subset = function(list, start, count) {
  if (typeof count !== 'undefined') {
    return list.slice(start, start + count);
  } else {
    return list.slice(start, list.length);
  }
};

module.exports = p5;

},{"../core/core":37}],74:[function(_dereq_,module,exports){
/**
 * @module Data
 * @submodule Conversion
 * @for p5
 * @requires core
 */

'use strict';

var p5 = _dereq_('../core/core');

/**
 * Converts a string to its floating point representation. The contents of a
 * string must resemble a number, or NaN (not a number) will be returned.
 * For example, float("1234.56") evaluates to 1234.56, but float("giraffe")
 * will return NaN.
 *
 * @method float
 * @param {String}  str float string to parse
 * @return {Number}     floating point representation of string
 * @example
 * <div><code>
 * var str = '20';
 * var diameter = float(str);
 * ellipse(width/2, height/2, diameter, diameter);
 * </code></div>
 */
p5.prototype.float = function(str) {
  return parseFloat(str);
};

/**
 * Converts a boolean, string, or float to its integer representation.
 * When an array of values is passed in, then an int array of the same length
 * is returned.
 *
 * @method int
 * @param {String|Boolean|Number|Array} n value to parse
 * @return {Number}                     integer representation of value
 * @example
 * <div class='norender'><code>
 * print(int("10")); // 10
 * print(int(10.31)); // 10
 * print(int(-10)); // -10
 * print(int(true)); // 1
 * print(int(false)); // 0
 * print(int([false, true, "10.3", 9.8])); // [0, 1, 10, 9]
 * </code></div>
 */
p5.prototype.int = function(n, radix) {
  if (typeof n === 'string') {
    radix = radix || 10;
    return parseInt(n, radix);
  } else if (typeof n === 'number') {
    return n | 0;
  } else if (typeof n === 'boolean') {
    return n ? 1 : 0;
  } else if (n instanceof Array) {
    return n.map(function(n) { return p5.prototype.int(n, radix); });
  }
};

/**
 * Converts a boolean, string or number to its string representation.
 * When an array of values is passed in, then an array of strings of the same
 * length is returned.
 *
 * @method str
 * @param {String|Boolean|Number|Array} n value to parse
 * @return {String}                     string representation of value
 * @example
 * <div class='norender'><code>
 * print(str("10"));  // "10"
 * print(str(10.31)); // "10.31"
 * print(str(-10));   // "-10"
 * print(str(true));  // "true"
 * print(str(false)); // "false"
 * print(str([true, "10.3", 9.8])); // [ "true", "10.3", "9.8" ]
 * </code></div>
 */
p5.prototype.str = function(n) {
  if (n instanceof Array) {
    return n.map(p5.prototype.str);
  } else {
    return String(n);
  }
};

/**
 * Converts a number or string to its boolean representation.
 * For a number, any non-zero value (positive or negative) evaluates to true,
 * while zero evaluates to false. For a string, the value "true" evaluates to
 * true, while any other value evaluates to false. When an array of number or
 * string values is passed in, then a array of booleans of the same length is
 * returned.
 *
 * @method boolean
 * @param {String|Boolean|Number|Array} n value to parse
 * @return {Boolean}                    boolean representation of value
 * @example
 * <div class='norender'><code>
 * print(boolean(0));               // false
 * print(boolean(1));               // true
 * print(boolean("true"));          // true
 * print(boolean("abcd"));          // false
 * print(boolean([0, 12, "true"])); // [false, true, false]
 * </code></div>
 */
p5.prototype.boolean = function(n) {
  if (typeof n === 'number') {
    return n !== 0;
  } else if (typeof n === 'string') {
    return n.toLowerCase() === 'true';
  } else if (typeof n === 'boolean') {
    return n;
  } else if (n instanceof Array) {
    return n.map(p5.prototype.boolean);
  }
};

/**
 * Converts a number, string or boolean to its byte representation.
 * A byte can be only a whole number between -128 and 127, so when a value
 * outside of this range is converted, it wraps around to the corresponding
 * byte representation. When an array of number, string or boolean values is
 * passed in, then an array of bytes the same length is returned.
 *
 * @method byte
 * @param {String|Boolean|Number|Array} n value to parse
 * @return {Number}                     byte representation of value
 * @example
 * <div class='norender'><code>
 * print(byte(127));               // 127
 * print(byte(128));               // -128
 * print(byte(23.4));              // 23
 * print(byte("23.4"));            // 23
 * print(byte(true));              // 1
 * print(byte([0, 255, "100"]));   // [0, -1, 100]
 * </code></div>
 */
p5.prototype.byte = function(n) {
  var nn = p5.prototype.int(n, 10);
  if (typeof nn === 'number') {
    return ((nn + 128) % 256) - 128;
  } else if (nn instanceof Array) {
    return nn.map(p5.prototype.byte);
  }
};

/**
 * Converts a number or string to its corresponding single-character
 * string representation. If a string parameter is provided, it is first
 * parsed as an integer and then translated into a single-character string.
 * When an array of number or string values is passed in, then an array of
 * single-character strings of the same length is returned.
 *
 * @method char
 * @param {String|Number|Array} n value to parse
 * @return {String}             string representation of value
 * @example
 * <div class='norender'><code>
 * print(char(65));                     // "A"
 * print(char("65"));                   // "A"
 * print(char([65, 66, 67]));           // [ "A", "B", "C" ]
 * print(join(char([65, 66, 67]), '')); // "ABC"
 * </code></div>
 */
p5.prototype.char = function(n) {
  if (typeof n === 'number' && !isNaN(n)) {
    return String.fromCharCode(n);
  } else if (n instanceof Array) {
    return n.map(p5.prototype.char);
  } else if (typeof n === 'string') {
    return p5.prototype.char(parseInt(n, 10));
  }
};

/**
 * Converts a single-character string to its corresponding integer
 * representation. When an array of single-character string values is passed
 * in, then an array of integers of the same length is returned.
 *
 * @method unchar
 * @param {String|Array} n value to parse
 * @return {Number}      integer representation of value
 * @example
 * <div class='norender'><code>
 * print(unchar("A"));               // 65
 * print(unchar(["A", "B", "C"]));   // [ 65, 66, 67 ]
 * print(unchar(split("ABC", "")));  // [ 65, 66, 67 ]
 * </code></div>
 */
p5.prototype.unchar = function(n) {
  if (typeof n === 'string' && n.length === 1) {
    return n.charCodeAt(0);
  } else if (n instanceof Array) {
    return n.map(p5.prototype.unchar);
  }
};

/**
 * Converts a number to a string in its equivalent hexadecimal notation. If a
 * second parameter is passed, it is used to set the number of characters to
 * generate in the hexadecimal notation. When an array is passed in, an
 * array of strings in hexadecimal notation of the same length is returned.
 *
 * @method hex
 * @param {Number|Array} n value to parse
 * @return {String}      hexadecimal string representation of value
 * @example
 * <div class='norender'><code>
 * print(hex(255));               // "000000FF"
 * print(hex(255, 6));            // "0000FF"
 * print(hex([0, 127, 255], 6));  // [ "000000", "00007F", "0000FF" ]
 * </code></div>
 */
p5.prototype.hex = function(n, digits) {
  digits = (digits === undefined || digits === null) ? digits = 8 : digits;
  if (n instanceof Array) {
    return n.map(function(n) { return p5.prototype.hex(n, digits); });
  } else if (typeof n === 'number') {
    if (n < 0) {
      n = 0xFFFFFFFF + n + 1;
    }
    var hex = Number(n).toString(16).toUpperCase();
    while (hex.length < digits) {
      hex = '0' + hex;
    }
    if (hex.length >= digits) {
      hex = hex.substring(hex.length - digits, hex.length);
    }
    return hex;
  }
};

/**
 * Converts a string representation of a hexadecimal number to its equivalent
 * integer value. When an array of strings in hexadecimal notation is passed
 * in, an array of integers of the same length is returned.
 *
 * @method unhex
 * @param {String|Array} n value to parse
 * @return {Number}      integer representation of hexadecimal value
 * @example
 * <div class='norender'><code>
 * print(unhex("A"));                // 10
 * print(unhex("FF"));               // 255
 * print(unhex(["FF", "AA", "00"])); // [ 255, 170, 0 ]
 * </code></div>
 */
p5.prototype.unhex = function(n) {
  if (n instanceof Array) {
    return n.map(p5.prototype.unhex);
  } else {
    return parseInt('0x' + n, 16);
  }
};

module.exports = p5;

},{"../core/core":37}],75:[function(_dereq_,module,exports){
/**
 * @module Data
 * @submodule String Functions
 * @for p5
 * @requires core
 */

'use strict';

var p5 = _dereq_('../core/core');

//return p5; //LM is this a mistake?

/**
 * Combines an array of Strings into one String, each separated by the
 * character(s) used for the separator parameter. To join arrays of ints or
 * floats, it's necessary to first convert them to Strings using nf() or
 * nfs().
 *
 * @method join
 * @param  {Array}  list      array of Strings to be joined
 * @param  {String} separator String to be placed between each item
 * @return {String}           joined String
 * @example
 * <div>
 * <code>
 * var array = ["Hello", "world!"]
 * var separator = " "
 * var message = join(array, separator);
 * text(message, 5, 50);
 * </code>
 * </div>
 */
p5.prototype.join = function(list, separator) {
  return list.join(separator);
};

/**
 * This function is used to apply a regular expression to a piece of text,
 * and return matching groups (elements found inside parentheses) as a
 * String array. If there are no matches, a null value will be returned.
 * If no groups are specified in the regular expression, but the sequence
 * matches, an array of length 1 (with the matched text as the first element
 * of the array) will be returned.
 * <br><br>
 * To use the function, first check to see if the result is null. If the
 * result is null, then the sequence did not match at all. If the sequence
 * did match, an array is returned.
 * <br><br>
 * If there are groups (specified by sets of parentheses) in the regular
 * expression, then the contents of each will be returned in the array.
 * Element [0] of a regular expression match returns the entire matching
 * string, and the match groups start at element [1] (the first group is [1],
 * the second [2], and so on).
 *
 * @method match
 * @param  {String} str    the String to be searched
 * @param  {String} regexp the regexp to be used for matching
 * @return {Array}         Array of Strings found
 * @example
 * <div>
 * <code>
 * var string = "Hello p5js*!"
 * var regexp = "p5js\\*"
 * var match = match(string, regexp);
 * text(match, 5, 50);
 * </code>
 * </div>
 */
p5.prototype.match =  function(str, reg) {
  return str.match(reg);
};

/**
 * This function is used to apply a regular expression to a piece of text,
 * and return a list of matching groups (elements found inside parentheses)
 * as a two-dimensional String array. If there are no matches, a null value
 * will be returned. If no groups are specified in the regular expression,
 * but the sequence matches, a two dimensional array is still returned, but
 * the second dimension is only of length one.
 * <br><br>
 * To use the function, first check to see if the result is null. If the
 * result is null, then the sequence did not match at all. If the sequence
 * did match, a 2D array is returned.
 * <br><br>
 * If there are groups (specified by sets of parentheses) in the regular
 * expression, then the contents of each will be returned in the array.
 * Assuming a loop with counter variable i, element [i][0] of a regular
 * expression match returns the entire matching string, and the match groups
 * start at element [i][1] (the first group is [i][1], the second [i][2],
 * and so on).
 *
 * @method matchAll
 * @param  {String} str    the String to be searched
 * @param  {String} regexp the regexp to be used for matching
 * @return {Array}         2d Array of Strings found
 * @example
 * <div class="norender">
 * <code>
 * var string = "Hello p5js*! Hello world!"
 * var regexp = "Hello"
 * matchAll(string, regexp);
 * </code>
 * </div>

 */
p5.prototype.matchAll = function(str, reg) {
  var re = new RegExp(reg, 'g');
  var match = re.exec(str);
  var matches = [];
  while (match !== null) {
    matches.push(match);
    // matched text: match[0]
    // match start: match.index
    // capturing group n: match[n]
    match = re.exec(str);
  }
  return matches;
};

/**
 * Utility function for formatting numbers into strings. There are two
 * versions: one for formatting floats, and one for formatting ints.
 * The values for the digits, left, and right parameters should always
 * be positive integers.
 *
 * @method nf
 * @param {Number|Array} num      the Number to format
 * @param {Number}       [left]   number of digits to the left of the
 *                                decimal point
 * @param {Number}       [right]  number of digits to the right of the
 *                                decimal point
 * @return {String|Array}         formatted String
 * @example
 * <div>
 * <code>
 * function setup() {
 *   background(200);
 *   var num = 112.53106115;
 *
 *   noStroke();
 *   fill(0);
 *   textSize(14);
 *   // Draw formatted numbers
 *   text(nf(num, 5, 2), 10, 20);
 *
 *   text(nf(num, 4, 3), 10, 55);
 *
 *   text(nf(num, 3, 6), 10, 85);
 *
 *   // Draw dividing lines
 *   stroke(120);
 *   line(0, 30, width, 30);
 *   line(0, 65, width, 65);
 * }
 * </code>
 * </div>
 */
p5.prototype.nf = function () {
  if (arguments[0] instanceof Array) {
    var a = arguments[1];
    var b = arguments[2];
    return arguments[0].map(function (x) {
      return doNf(x, a, b);
    });
  }
  else{
    var typeOfFirst = Object.prototype.toString.call(arguments[0]);
    if(typeOfFirst === '[object Arguments]'){
      if(arguments[0].length===3){
        return this.nf(arguments[0][0],arguments[0][1],arguments[0][2]);
      }
      else if(arguments[0].length===2){
        return this.nf(arguments[0][0],arguments[0][1]);
      }
      else{
        return this.nf(arguments[0][0]);
      }
    }
    else {
      return doNf.apply(this, arguments);
    }
  }
};

function doNf() {
  var num = arguments[0];
  var neg = num < 0;
  var n = neg ? num.toString().substring(1) : num.toString();
  var decimalInd = n.indexOf('.');
  var intPart = decimalInd !== -1 ? n.substring(0, decimalInd) : n;
  var decPart = decimalInd !== -1 ? n.substring(decimalInd + 1) : '';
  var str = neg ? '-' : '';
  if (arguments.length === 3) {
    var decimal = '';
    if(decimalInd !== -1 || arguments[2] - decPart.length > 0){
      decimal = '.';
    }
    if (decPart.length > arguments[2]) {
      decPart = decPart.substring(0, arguments[2]);
    }
    for (var i = 0; i < arguments[1] - intPart.length; i++) {
      str += '0';
    }
    str += intPart;
    str += decimal;
    str += decPart;
    for (var j = 0; j < arguments[2] - decPart.length; j++) {
      str += '0';
    }
    return str;
  }
  else {
    for (var k = 0; k < Math.max(arguments[1] - intPart.length, 0); k++) {
      str += '0';
    }
    str += n;
    return str;
  }
}

/**
 * Utility function for formatting numbers into strings and placing
 * appropriate commas to mark units of 1000. There are two versions: one
 * for formatting ints, and one for formatting an array of ints. The value
 * for the right parameter should always be a positive integer.
 *
 * @method nfc
 * @param  {Number|Array}   num     the Number to format
 * @param  {Number}         [right] number of digits to the right of the
 *                                  decimal point
 * @return {String|Array}           formatted String
 * @example
 * <div>
 * <code>
 * function setup() {
 *   background(200);
 *   var num = 11253106.115;
 *   var numArr = new Array(1,1,2);
 *
 *   noStroke();
 *   fill(0);
 *   textSize(12);
 *
 *   // Draw formatted numbers
 *   text(nfc(num, 4, 2), 10, 30);
 *   text(nfc(numArr, 2, 1), 10, 80);
 *
 *   // Draw dividing line
 *   stroke(120);
 *   line(0, 50, width, 50);
 * }
 * </code>
 * </div>
 */
p5.prototype.nfc = function () {
  if (arguments[0] instanceof Array) {
    var a = arguments[1];
    return arguments[0].map(function (x) {
      return doNfc(x, a);
    });
  } else {
    return doNfc.apply(this, arguments);
  }
};
function doNfc() {
  var num = arguments[0].toString();
  var dec = num.indexOf('.');
  var rem = dec !== -1 ? num.substring(dec) : '';
  var n = dec !== -1 ? num.substring(0, dec) : num;
  n = n.toString().replace(/\B(?=(\d{3})+(?!\d))/g, ',');
  if (arguments[1] === 0) {
    rem = '';
  }
  else if(arguments[1] !== undefined){
    if(arguments[1] > rem.length){
      rem+= dec === -1 ? '.' : '';
      var len = arguments[1] - rem.length + 1;
      for(var i =0; i< len; i++){
        rem += '0';
      }
    }
    else{
      rem = rem.substring(0, arguments[1] + 1);
    }
  }
  return n + rem;
}

/**
 * Utility function for formatting numbers into strings. Similar to nf() but
 * puts a "+" in front of positive numbers and a "-" in front of negative
 * numbers. There are two versions: one for formatting floats, and one for
 * formatting ints. The values for left, and right parameters
 * should always be positive integers.
 *
 * @method nfp
 * @param {Number|Array} num      the Number to format
 * @param {Number}       [left]   number of digits to the left of the decimal
 *                                point
 * @param {Number}       [right]  number of digits to the right of the
 *                                decimal point
 * @return {String|Array}         formatted String
 * @example
 * <div>
 * <code>
 * function setup() {
 *   background(200);
 *   var num1 = 11253106.115;
 *   var num2 = -11253106.115;
 *
 *   noStroke();
 *   fill(0);
 *   textSize(12);
 *
 *   // Draw formatted numbers
 *   text(nfp(num1, 4, 2), 10, 30);
 *   text(nfp(num2, 4, 2), 10, 80);
 *
 *   // Draw dividing line
 *   stroke(120);
 *   line(0, 50, width, 50);
 * }
 * </code>
 * </div>
 */
p5.prototype.nfp = function() {
  var nfRes = this.nf.apply(this, arguments);
  if (nfRes instanceof Array) {
    return nfRes.map(addNfp);
  } else {
    return addNfp(nfRes);
  }
};

function addNfp() {
  return (
    parseFloat(arguments[0]) > 0) ?
    '+'+arguments[0].toString() :
    arguments[0].toString();
}

/**
 * Utility function for formatting numbers into strings. Similar to nf() but
 * puts a " " (space) in front of positive numbers and a "-" in front of
 * negative numbers. There are two versions: one for formatting floats, and
 * one for formatting ints. The values for the digits, left, and right
 * parameters should always be positive integers.
 *
 * @method nfs
 * @param {Number|Array} num      the Number to format
 * @param {Number}       [left]   number of digits to the left of the decimal
 *                                point
 * @param {Number}       [right]  number of digits to the right of the
 *                                decimal point
 * @return {String|Array}         formatted String
 * @example
 * <div>
 * <code>
 * function setup() {
 *   background(200);
 *   var num1 = 11253106.115;
 *   var num2 = -11253106.115;
 *
 *   noStroke();
 *   fill(0);
 *   textSize(12);
 *   // Draw formatted numbers
 *   text(nfs(num1, 4, 2), 10, 30);
 *
 *   text(nfs(num2, 4, 2), 10, 80);
 *
 *   // Draw dividing line
 *   stroke(120);
 *   line(0, 50, width, 50);
 * }
 * </code>
 * </div>
 */
p5.prototype.nfs = function() {
  var nfRes = this.nf.apply(this, arguments);
  if (nfRes instanceof Array) {
    return nfRes.map(addNfs);
  } else {
    return addNfs(nfRes);
  }
};

function addNfs() {
  return (
    parseFloat(arguments[0]) > 0) ?
    ' '+arguments[0].toString() :
    arguments[0].toString();
}

/**
 * The split() function maps to String.split(), it breaks a String into
 * pieces using a character or string as the delimiter. The delim parameter
 * specifies the character or characters that mark the boundaries between
 * each piece. A String[] array is returned that contains each of the pieces.
 *
 * The splitTokens() function works in a similar fashion, except that it
 * splits using a range of characters instead of a specific character or
 * sequence.
 *
 * @method split
 * @param  {String} value the String to be split
 * @param  {String} delim the String used to separate the data
 * @return {Array}  Array of Strings
 * @example
 * <div>
 * <code>
 * var names = "Pat,Xio,Alex"
 * var splitString = split(names, ",");
 * text(splitString[0], 5, 30);
 * text(splitString[1], 5, 50);
 * text(splitString[2], 5, 70);
 * </code>
 * </div>
 */
p5.prototype.split = function(str, delim) {
  return str.split(delim);
};

/**
 * The splitTokens() function splits a String at one or many character
 * delimiters or "tokens." The delim parameter specifies the character or
 * characters to be used as a boundary.
 * <br><br>
 * If no delim characters are specified, any whitespace character is used to
 * split. Whitespace characters include tab (\t), line feed (\n), carriage
 * return (\r), form feed (\f), and space.
 *
 * @method splitTokens
 * @param  {String} value   the String to be split
 * @param  {String} [delim] list of individual Strings that will be used as
 *                          separators
 * @return {Array}          Array of Strings
 * @example
 * <div class = "norender">
 * <code>
 * function setup() {
 *   var myStr = "Mango, Banana, Lime";
 *   var myStrArr = splitTokens(myStr, ",");
 *
 *   print(myStrArr); // prints : ["Mango"," Banana"," Lime"]
 * }
 * </code>
 * </div>
 */
p5.prototype.splitTokens = function() {
  var d,sqo,sqc,str;
  str = arguments[1];
  if (arguments.length > 1) {
    sqc = /\]/g.exec(str);
    sqo = /\[/g.exec(str);
    if ( sqo && sqc ) {
      str = str.slice(0, sqc.index) + str.slice(sqc.index+1);
      sqo = /\[/g.exec(str);
      str = str.slice(0, sqo.index) + str.slice(sqo.index+1);
      d = new RegExp('[\\['+str+'\\]]','g');
    } else if ( sqc ) {
      str = str.slice(0, sqc.index) + str.slice(sqc.index+1);
      d = new RegExp('[' + str + '\\]]', 'g');
    } else if(sqo) {
      str = str.slice(0, sqo.index) + str.slice(sqo.index+1);
      d = new RegExp('[' + str + '\\[]', 'g');
    } else {
      d = new RegExp('[' + str + ']', 'g');
    }
  } else {
    d = /\s/g;
  }
  return arguments[0].split(d).filter(function(n){return n;});
};

/**
 * Removes whitespace characters from the beginning and end of a String. In
 * addition to standard whitespace characters such as space, carriage return,
 * and tab, this function also removes the Unicode "nbsp" character.
 *
 * @method trim
 * @param  {String|Array} str a String or Array of Strings to be trimmed
 * @return {String|Array}       a trimmed String or Array of Strings
 * @example
 * <div>
 * <code>
 * var string = trim("  No new lines\n   ");
 * text(string +" here", 2, 50);
 * </code>
 * </div>
 */
p5.prototype.trim = function(str) {
  if (str instanceof Array) {
    return str.map(this.trim);
  } else {
    return str.trim();
  }
};

module.exports = p5;

},{"../core/core":37}],76:[function(_dereq_,module,exports){
/**
 * @module IO
 * @submodule Time & Date
 * @for p5
 * @requires core
 */

'use strict';

var p5 = _dereq_('../core/core');

/**
 * p5.js communicates with the clock on your computer. The day() function
 * returns the current day as a value from 1 - 31.
 *
 * @method day
 * @return {Number} the current day
 * @example
 * <div>
 * <code>
 * var d = day();
 * text("Current day: \n" + d, 5, 50);
 * </code>
 * </div>
 */
p5.prototype.day = function() {
  return new Date().getDate();
};

/**
 * p5.js communicates with the clock on your computer. The hour() function
 * returns the current hour as a value from 0 - 23.
 *
 * @method hour
 * @return {Number} the current hour
 * @example
 * <div>
 * <code>
 * var h = hour();
 * text("Current hour:\n" + h, 5, 50);
 * </code>
 * </div>
 */
p5.prototype.hour = function() {
  return new Date().getHours();
};

/**
 * p5.js communicates with the clock on your computer. The minute() function
 * returns the current minute as a value from 0 - 59.
 *
 * @method minute
 * @return {Number} the current minute
 * @example
 * <div>
 * <code>
 * var m = minute();
 * text("Current minute: \n" + m, 5, 50);
 * </code>
 * </div>
 */
p5.prototype.minute = function() {
  return new Date().getMinutes();
};

/**
 * Returns the number of milliseconds (thousandths of a second) since
 * starting the program. This information is often used for timing events and
 * animation sequences.
 *
 * @method millis
 * @return {Number} the number of milliseconds since starting the program
 * @example
 * <div>
 * <code>
 * var millisecond = millis();
 * text("Milliseconds \nrunning: \n" + millisecond, 5, 40);
 * </code>
 * </div>
 */
p5.prototype.millis = function() {
  return window.performance.now();
};

/**
 * p5.js communicates with the clock on your computer. The month() function
 * returns the current month as a value from 1 - 12.
 *
 * @method month
 * @return {Number} the current month
 * @example
 * <div>
 * <code>
 * var m = month();
 * text("Current month: \n" + m, 5, 50);
 * </code>
 * </div>
 */
p5.prototype.month = function() {
  return new Date().getMonth() + 1; //January is 0!
};

/**
 * p5.js communicates with the clock on your computer. The second() function
 * returns the current second as a value from 0 - 59.
 *
 * @method second
 * @return {Number} the current second
 * @example
 * <div>
 * <code>
 * var s = second();
 * text("Current second: \n" + s, 5, 50);
 * </code>
 * </div>
 */
p5.prototype.second = function() {
  return new Date().getSeconds();
};

/**
 * p5.js communicates with the clock on your computer. The year() function
 * returns the current year as an integer (2014, 2015, 2016, etc).
 *
 * @method year
 * @return {Number} the current year
 * @example
 * <div>
 * <code>
 * var y = year();
 * text("Current year: \n" + y, 5, 50);
 * </code>
 * </div>
 */
p5.prototype.year = function() {
  return new Date().getFullYear();
};

module.exports = p5;

},{"../core/core":37}],77:[function(_dereq_,module,exports){
/**
 * @module Lights, Camera
 * @submodule Camera
 * @for p5
 * @requires core
 */

'use strict';

var p5 = _dereq_('../core/core');

/**
 * Sets camera position
 * @method camera
 * @param  {Number} x  camera position value on x axis
 * @param  {Number} y  camera position value on y axis
 * @param  {Number} z  camera position value on z axis
 * @return {p5}        the p5 object
 * @example
 * <div>
 * <code>
 * function setup(){
 *   createCanvas(100, 100, WEBGL);
 * }
 * function draw(){
 *  //move the camera away from the plane by a sin wave
 *  camera(0, 0, sin(frameCount * 0.01) * 100);
 *  plane(120, 120);
 * }
 * </code>
 * </div>
 */
p5.prototype.camera = function(x, y, z){
  var args = new Array(arguments.length);
  for (var i = 0; i < args.length; ++i) {
    args[i] = arguments[i];
  }
  this._validateParameters(
    'camera',
    args,
    ['Number', 'Number', 'Number']
  );
  //what it manipulates is the model view matrix
  this._renderer.translate(-x, -y, -z);
};

/**
 * Sets perspective camera
 * @method  perspective
 * @param  {Number} fovy   camera frustum vertical field of view,
 *                         from bottom to top of view, in degrees
 * @param  {Number} aspect camera frustum aspect ratio
 * @param  {Number} near   frustum near plane length
 * @param  {Number} far    frustum far plane length
 * @return {p5}            the p5 object
 * @example
 * <div>
 * <code>
 * //drag mouse to toggle the world!
 * //you will see there's a vanish point
 * function setup(){
 *   createCanvas(100, 100, WEBGL);
 *   perspective(60 / 180 * PI, width/height, 0.1, 100);
 * }
 * function draw(){
 *  background(200);
 *  orbitControl();
 *  for(var i = -1; i < 2; i++){
 *     for(var j = -2; j < 3; j++){
 *       push();
 *       translate(i*160, 0, j*160);
 *       box(40, 40, 40);
 *       pop();
 *     }
 *   }
 * }
 * </code>
 * </div>
 */
p5.prototype.perspective = function(fovy,aspect,near,far) {
  var args = new Array(arguments.length);
  for (var i = 0; i < args.length; ++i) {
    args[i] = arguments[i];
  }
  this._validateParameters(
    'perspective',
    args,
    ['Number', 'Number', 'Number', 'Number']
  );
  this._renderer.uPMatrix = p5.Matrix.identity();
  this._renderer.uPMatrix.perspective(fovy,aspect,near,far);
  this._renderer._isSetCamera = true;
};

/**
 * Setup ortho camera
 * @method  ortho
 * @param  {Number} left   camera frustum left plane
 * @param  {Number} right  camera frustum right plane
 * @param  {Number} bottom camera frustum bottom plane
 * @param  {Number} top    camera frustum top plane
 * @param  {Number} near   camera frustum near plane
 * @param  {Number} far    camera frustum far plane
 * @return {p5}            the p5 object
 * @example
 * <div>
 * <code>
 * //drag mouse to toggle the world!
 * //there's no vanish point
 * function setup(){
 *   createCanvas(100, 100, WEBGL);
 *   ortho(-width/2, width/2, height/2, -height/2, 0.1, 100);
 * }
 * function draw(){
 *  background(200);
 *  orbitControl();
 *  for(var i = -1; i < 2; i++){
 *     for(var j = -2; j < 3; j++){
 *       push();
 *       translate(i*160, 0, j*160);
 *       box(40, 40, 40);
 *       pop();
 *     }
 *   }
 * }
 * </code>
 * </div>
 */
p5.prototype.ortho = function(left,right,bottom,top,near,far) {
  var args = new Array(arguments.length);
  for (var i = 0; i < args.length; ++i) {
    args[i] = arguments[i];
  }
  this._validateParameters(
    'ortho',
    args,
      ['Number', 'Number', 'Number', 'Number', 'Number', 'Number']
  );
  left /= this.width;
  right /= this.width;
  top /= this.height;
  bottom /= this.height;
  this._renderer.uPMatrix = p5.Matrix.identity();
  this._renderer.uPMatrix.ortho(left,right,bottom,top,near,far);
  this._renderer._isSetCamera = true;
};

module.exports = p5;

},{"../core/core":37}],78:[function(_dereq_,module,exports){
'use strict';

var p5 = _dereq_('../core/core');

//@TODO: implement full orbit controls including
//pan, zoom, quaternion rotation, etc.
p5.prototype.orbitControl = function(){
  if(this.mouseIsPressed){
    this.rotateY((this.mouseX - this.width / 2) / (this.width / 2));
    this.rotateX((this.mouseY - this.height / 2) / (this.width / 2));
  }
  return this;
};

module.exports = p5;
},{"../core/core":37}],79:[function(_dereq_,module,exports){
/**
 * @module Lights, Camera
 * @submodule Lights
 * @for p5
 * @requires core
 */

'use strict';

var p5 = _dereq_('../core/core');

/**
 * Creates an ambient light with a color
 * @method  ambientLight
 * @param  {Number|Array|String|p5.Color} v1  gray value,
 * red or hue value (depending on the current color mode),
 * or color Array, or CSS color string
 * @param  {Number}            [v2] optional: green or saturation value
 * @param  {Number}            [v3] optional: blue or brightness value
 * @param  {Number}            [a]  optional: opacity
 * @return {p5}                the p5 object
 * @example
 * <div>
 * <code>
 * function setup(){
 *   createCanvas(100, 100, WEBGL);
 * }
 * function draw(){
 *   background(0);
 *   ambientLight(150);
 *   ambientMaterial(250);
 *   sphere(200);
 * }
 * </code>
 * </div>
 */
p5.prototype.ambientLight = function(v1, v2, v3, a){
  var gl = this._renderer.GL;
  var shaderProgram = this._renderer._getShader(
    'lightVert', 'lightTextureFrag');

  gl.useProgram(shaderProgram);
  shaderProgram.uAmbientColor = gl.getUniformLocation(
    shaderProgram,
    'uAmbientColor[' + this._renderer.ambientLightCount + ']');

  var color = this._renderer._pInst.color.apply(
    this._renderer._pInst, arguments);
  var colors = color._array;

  gl.uniform3f( shaderProgram.uAmbientColor,
    colors[0], colors[1], colors[2]);

  //in case there's no material color for the geometry
  shaderProgram.uMaterialColor = gl.getUniformLocation(
    shaderProgram, 'uMaterialColor' );
  gl.uniform4f( shaderProgram.uMaterialColor, 1, 1, 1, 1);

  this._renderer.ambientLightCount ++;
  shaderProgram.uAmbientLightCount =
    gl.getUniformLocation(shaderProgram, 'uAmbientLightCount');
  gl.uniform1i(shaderProgram.uAmbientLightCount,
    this._renderer.ambientLightCount);

  return this;
};

/**
 * Creates a directional light with a color and a direction
 * @method  directionalLight
 * @param  {Number|Array|String|p5.Color} v1   gray value,
 * red or hue value (depending on the current color mode),
 * or color Array, or CSS color string
 * @param  {Number}          [v2] optional: green or saturation value
 * @param  {Number}          [v3] optional: blue or brightness value
 * @param  {Number}          [a]  optional: opacity
 * @param  {Number|p5.Vector} x   x axis direction or a p5.Vector
 * @param  {Number}          [y]  optional: y axis direction
 * @param  {Number}          [z]  optional: z axis direction
 * @return {p5}              the p5 object
 * @example
 * <div>
 * <code>
 * function setup(){
 *   createCanvas(100, 100, WEBGL);
 * }
 * function draw(){
 *   background(0);
 *   //move your mouse to change light direction
 *   var dirX = (mouseX / width - 0.5) *2;
 *   var dirY = (mouseY / height - 0.5) *(-2);
 *   directionalLight(250, 250, 250, dirX, dirY, 0.25);
 *   ambientMaterial(250);
 *   sphere(200);
 * }
 * </code>
 * </div>
 */
p5.prototype.directionalLight = function(v1, v2, v3, a, x, y, z) {
  // TODO(jgessner): Find an example using this and profile it.
  // var args = new Array(arguments.length);
  // for (var i = 0; i < args.length; ++i) {
  //   args[i] = arguments[i];
  // }
  // this._validateParameters(
  //   'directionalLight',
  //   args,
  //   [
  //     //rgbaxyz
  //     ['Number', 'Number', 'Number', 'Number', 'Number', 'Number', 'Number'],
  //     //rgbxyz
  //     ['Number', 'Number', 'Number', 'Number', 'Number', 'Number'],
  //     //caxyz
  //     ['Number', 'Number', 'Number', 'Number', 'Number'],
  //     //cxyz
  //     ['Number', 'Number', 'Number', 'Number'],
  //     ['String', 'Number', 'Number', 'Number'],
  //     ['Array', 'Number', 'Number', 'Number'],
  //     ['Object', 'Number', 'Number', 'Number'],
  //     //rgbavector
  //     ['Number', 'Number', 'Number', 'Number', 'Object'],
  //     //rgbvector
  //     ['Number', 'Number', 'Number', 'Object'],
  //     //cavector
  //     ['Number', 'Number', 'Object'],
  //     //cvector
  //     ['Number', 'Object'],
  //     ['String', 'Object'],
  //     ['Array', 'Object'],
  //     ['Object', 'Object']
  //   ]
  // );

  var gl = this._renderer.GL;
  var shaderProgram = this._renderer._getShader(
    'lightVert', 'lightTextureFrag');

  gl.useProgram(shaderProgram);
  shaderProgram.uDirectionalColor = gl.getUniformLocation(
    shaderProgram,
    'uDirectionalColor[' + this._renderer.directionalLightCount + ']');

  //@TODO: check parameters number
  var color = this._renderer._pInst.color.apply(
    this._renderer._pInst, [v1, v2, v3]);
  var colors = color._array;

  gl.uniform3f( shaderProgram.uDirectionalColor,
    colors[0], colors[1], colors[2]);

  var _x, _y, _z;

  var args = new Array(arguments.length);
  for (var i = 0; i < args.length; ++i) {
    args[i] = arguments[i];
  }
  if(typeof args[args.length-1] === 'number'){
    _x = args[args.length-3];
    _y = args[args.length-2];
    _z = args[args.length-1];

  }else{
    try{
      _x = args[args.length-1].x;
      _y = args[args.length-1].y;
      _z = args[args.length-1].z;
    }
    catch(error){
      throw error;
    }
  }

  shaderProgram.uLightingDirection = gl.getUniformLocation(
    shaderProgram,
    'uLightingDirection[' + this._renderer.directionalLightCount + ']');
  gl.uniform3f( shaderProgram.uLightingDirection, _x, _y, _z);

  //in case there's no material color for the geometry
  shaderProgram.uMaterialColor = gl.getUniformLocation(
    shaderProgram, 'uMaterialColor' );
  gl.uniform4f( shaderProgram.uMaterialColor, 1, 1, 1, 1);

  this._renderer.directionalLightCount ++;
  shaderProgram.uDirectionalLightCount =
    gl.getUniformLocation(shaderProgram, 'uDirectionalLightCount');
  gl.uniform1i(shaderProgram.uDirectionalLightCount,
    this._renderer.directionalLightCount);

  return this;
};

/**
 * Creates a point light with a color and a light position
 * @method  pointLight
 * @param  {Number|Array|String|p5.Color} v1   gray value,
 * red or hue value (depending on the current color mode),
 * or color Array, or CSS color string
 * @param  {Number}          [v2] optional: green or saturation value
 * @param  {Number}          [v3] optional: blue or brightness value
 * @param  {Number}          [a]  optional: opacity
 * @param  {Number|p5.Vector} x   x axis position or a p5.Vector
 * @param  {Number}          [y]  optional: y axis position
 * @param  {Number}          [z]  optional: z axis position
 * @return {p5}              the p5 object
 * @example
 * <div>
 * <code>
 * function setup(){
 *   createCanvas(100, 100, WEBGL);
 * }
 * function draw(){
 *   background(0);
 *   //move your mouse to change light position
 *   var locY = (mouseY / height - 0.5) *(-2);
 *   var locX = (mouseX / width - 0.5) *2;
 *   //to set the light position,
 *   //think of the world's coordinate as:
 *   // -1,1 -------- 1,1
 *   //   |            |
 *   //   |            |
 *   //   |            |
 *   // -1,-1---------1,-1
 *   pointLight(250, 250, 250, locX, locY, 0);
 *   ambientMaterial(250);
 *   sphere(200);
 * }
 * </code>
 * </div>
 */
p5.prototype.pointLight = function(v1, v2, v3, a, x, y, z) {
  // TODO(jgessner): Find an example using this and profile it.
  // var args = new Array(arguments.length);
  // for (var i = 0; i < args.length; ++i) {
  //   args[i] = arguments[i];
  // }
  // this._validateParameters(
  //   'pointLight',
  //   arguments,
  //   [
  //     //rgbaxyz
  //     ['Number', 'Number', 'Number', 'Number', 'Number', 'Number', 'Number'],
  //     //rgbxyz
  //     ['Number', 'Number', 'Number', 'Number', 'Number', 'Number'],
  //     //caxyz
  //     ['Number', 'Number', 'Number', 'Number', 'Number'],
  //     //cxyz
  //     ['Number', 'Number', 'Number', 'Number'],
  //     ['String', 'Number', 'Number', 'Number'],
  //     ['Array', 'Number', 'Number', 'Number'],
  //     ['Object', 'Number', 'Number', 'Number'],
  //     //rgbavector
  //     ['Number', 'Number', 'Number', 'Number', 'Object'],
  //     //rgbvector
  //     ['Number', 'Number', 'Number', 'Object'],
  //     //cavector
  //     ['Number', 'Number', 'Object'],
  //     //cvector
  //     ['Number', 'Object'],
  //     ['String', 'Object'],
  //     ['Array', 'Object'],
  //     ['Object', 'Object']
  //   ]
  // );

  var gl = this._renderer.GL;
  var shaderProgram = this._renderer._getShader(
    'lightVert', 'lightTextureFrag');

  gl.useProgram(shaderProgram);
  shaderProgram.uPointLightColor = gl.getUniformLocation(
    shaderProgram,
    'uPointLightColor[' + this._renderer.pointLightCount + ']');

  //@TODO: check parameters number
  var color = this._renderer._pInst.color.apply(
    this._renderer._pInst, [v1, v2, v3]);
  var colors = color._array;

  gl.uniform3f( shaderProgram.uPointLightColor,
    colors[0], colors[1], colors[2]);

  var _x, _y, _z;

  var args = new Array(arguments.length);
  for (var i = 0; i < args.length; ++i) {
    args[i] = arguments[i];
  }
  if(typeof args[args.length-1] === 'number'){
    _x = args[args.length-3];
    _y = args[args.length-2];
    _z = args[args.length-1];

  }else{
    try{
      _x = args[args.length-1].x;
      _y = args[args.length-1].y;
      _z = args[args.length-1].z;
    }
    catch(error){
      throw error;
    }
  }

  shaderProgram.uPointLightLocation = gl.getUniformLocation(
    shaderProgram,
    'uPointLightLocation[' + this._renderer.pointLightCount + ']');
  gl.uniform3f( shaderProgram.uPointLightLocation, _x, _y, _z);

  //in case there's no material color for the geometry
  shaderProgram.uMaterialColor = gl.getUniformLocation(
    shaderProgram, 'uMaterialColor' );
  gl.uniform4f( shaderProgram.uMaterialColor, 1, 1, 1, 1);

  this._renderer.pointLightCount ++;
  shaderProgram.uPointLightCount =
    gl.getUniformLocation(shaderProgram, 'uPointLightCount');
  gl.uniform1i(shaderProgram.uPointLightCount,
    this._renderer.pointLightCount);

  return this;
};

module.exports = p5;

},{"../core/core":37}],80:[function(_dereq_,module,exports){
/**
 * @module Shape
 * @submodule 3D Models
 * @for p5
 * @requires core
 * @requires p5.Geometry3D
 */

'use strict';

var p5 = _dereq_('../core/core');
_dereq_('./p5.Geometry');

/**
 * Load a 3d model from an OBJ file.
 * <br><br>
 * One of the limitations of the OBJ format is that it doesn't have a built-in
 * sense of scale. This means that models exported from different programs might
 * be very different sizes. If your model isn't displaying, try calling
 * loadModel() with the normalized parameter set to true. This will resize the
 * model to a scale appropriate for p5. You can also make additional changes to
 * the final size of your model with the scale() function.
 *
 * @method loadModel
 * @param  {String} path Path of the model to be loaded
 * @param  {Boolean} [normalize] If true, scale the model to a
 *                                standardized size when loading
 * @param  {Function(p5.Geometry3D)} [successCallback] Function to be called
 *                                   once the model is loaded. Will be passed
 *                                   the 3D model object.
 * @param  {Function(Event)}    [failureCallback] called with event error if
 *                                the image fails to load.
 * @return {p5.Geometry} the p5.Geometry3D object
 * @example
 * <div>
 * <code>
 * //draw a spinning teapot
 * var teapot;
 *
 * function setup(){
 *   createCanvas(100, 100, WEBGL);
 *
 *   teapot = loadModel('assets/teapot.obj');
 * }
 *
 * function draw(){
 *   background(200);
 *   rotateX(frameCount * 0.01);
 *   rotateY(frameCount * 0.01);
 *   model(teapot);
 * }
 * </code>
 * </div>
 */
p5.prototype.loadModel = function () {
  var path = arguments[0];
  var normalize;
  var successCallback;
  var failureCallback;
  if(typeof arguments[1] === 'boolean') {
    normalize = arguments[1];
    successCallback = arguments[2];
    failureCallback = arguments[3];
  } else {
    normalize = false;
    successCallback = arguments[1];
    failureCallback = arguments[2];
  }

  var model = new p5.Geometry();
  model.gid = path + '|' + normalize;
  this.loadStrings(path, function(strings) {
    parseObj(model, strings);

    if (normalize) {
      model.normalize();
    }

    if (typeof successCallback === 'function') {
      successCallback(model);
    }
  }.bind(this), failureCallback);

  return model;
};

/**
 * Parse OBJ lines into model. For reference, this is what a simple model of a
 * square might look like:
 *
 * v -0.5 -0.5 0.5
 * v -0.5 -0.5 -0.5
 * v -0.5 0.5 -0.5
 * v -0.5 0.5 0.5
 *
 * f 4 3 2 1
 */
function parseObj( model, lines ) {
  // OBJ allows a face to specify an index for a vertex (in the above example),
  // but it also allows you to specify a custom combination of vertex, UV
  // coordinate, and vertex normal. So, "3/4/3" would mean, "use vertex 3 with
  // UV coordinate 4 and vertex normal 3". In WebGL, every vertex with different
  // parameters must be a different vertex, so loadedVerts is used to
  // temporarily store the parsed vertices, normals, etc., and indexedVerts is
  // used to map a specific combination (keyed on, for example, the string
  // "3/4/3"), to the actual index of the newly created vertex in the final
  // object.
  var loadedVerts = {'v' : [],
                     'vt' : [],
                     'vn' : []},
      indexedVerts = {};

  for (var line = 0; line < lines.length; ++line) {
    // Each line is a separate object (vertex, face, vertex normal, etc)
    // For each line, split it into tokens on whitespace. The first token
    // describes the type.
    var tokens = lines[line].trim().split(/\b\s+/);

    if (tokens.length > 0) {
      if (tokens[0] === 'v' || tokens[0] === 'vn') {
        // Check if this line describes a vertex or vertex normal.
        // It will have three numeric parameters.
        var vertex = new p5.Vector(parseFloat(tokens[1]),
                                   parseFloat(tokens[2]),
                                   parseFloat(tokens[3]));
        loadedVerts[tokens[0]].push(vertex);
      } else if (tokens[0] === 'vt') {
        // Check if this line describes a texture coordinate.
        // It will have two numeric parameters.
        var texVertex = [parseFloat(tokens[1]), parseFloat(tokens[2])];
        loadedVerts[tokens[0]].push(texVertex);
      } else if (tokens[0] === 'f') {
        // Check if this line describes a face.
        // OBJ faces can have more than three points. Triangulate points.
        for (var tri = 3; tri < tokens.length; ++tri) {
          var face = [];

          var vertexTokens = [1, tri - 1, tri];

          for (var tokenInd = 0; tokenInd < vertexTokens.length; ++tokenInd) {
            // Now, convert the given token into an index
            var vertString = tokens[vertexTokens[tokenInd]];
            var vertIndex = 0;

            // TODO: Faces can technically use negative numbers to refer to the
            // previous nth vertex. I haven't seen this used in practice, but
            // it might be good to implement this in the future.

            if (indexedVerts[vertString] !== undefined) {
              vertIndex = indexedVerts[vertString];
            } else {
              var vertParts = vertString.split('/');
              for (var i = 0; i < vertParts.length; i++) {
                vertParts[i] = parseInt(vertParts[i]) - 1;
              }

              vertIndex = indexedVerts[vertString] = model.vertices.length;
              model.vertices.push(loadedVerts.v[vertParts[0]].copy());
              if (loadedVerts.vt[vertParts[1]]) {
                model.uvs.push(loadedVerts.vt[vertParts[1]].slice());
              } else {
                model.uvs.push([0, 0]);
              }

              if (loadedVerts.vn[vertParts[2]]) {
                model.vertexNormals.push(loadedVerts.vn[vertParts[2]].copy());
              }
            }

            face.push(vertIndex);
          }

          model.faces.push(face);
        }
      }
    }
  }

  // If the model doesn't have normals, compute the normals
  if(model.vertexNormals.length === 0) {
    model.computeNormals();
  }

  return model;
}

/**
 * Render a 3d model to the screen.
 *
 * @method model
 * @param  {p5.Geometry} model Loaded 3d model to be rendered
 * @example
 * <div>
 * <code>
 * //draw a spinning teapot
 * var teapot;
 *
 * function setup(){
 *   createCanvas(100, 100, WEBGL);
 *
 *   teapot = loadModel('assets/teapot.obj');
 * }
 *
 * function draw(){
 *   background(200);
 *   rotateX(frameCount * 0.01);
 *   rotateY(frameCount * 0.01);
 *   model(teapot);
 * }
 * </code>
 * </div>
 */
p5.prototype.model = function ( model ) {
  if (model.vertices.length > 0) {
    if (!this._renderer.geometryInHash(model.gid)) {
      this._renderer.createBuffers(model.gid, model);
    }

    this._renderer.drawBuffers(model.gid);
  }
};

module.exports = p5;

},{"../core/core":37,"./p5.Geometry":82}],81:[function(_dereq_,module,exports){
/**
 * @module Lights, Camera
 * @submodule Material
 * @for p5
 * @requires core
 */

'use strict';

var p5 = _dereq_('../core/core');
//require('./p5.Texture');

/**
 * Normal material for geometry
 * @method normalMaterial
 * @return {p5}                the p5 object
 * @example
 * <div>
 * <code>
 * function setup(){
 *   createCanvas(100, 100, WEBGL);
 * }
 *
 * function draw(){
 *  background(0);
 *  normalMaterial();
 *  sphere(200);
 * }
 * </code>
 * </div>
 */
p5.prototype.normalMaterial = function(){
  this._renderer._getShader('normalVert', 'normalFrag');
  return this;
};

/**
 * Texture for geometry
 * @method texture
 * @param {p5.Image | p5.MediaElement | p5.Graphics} tex 2-dimensional graphics
 *                    to render as texture
 * @return {p5}                the p5 object
 * @example
 * <div>
 * <code>
 * var img;
 * function setup(){
 *   createCanvas(100, 100, WEBGL);
 *   img = loadImage("assets/laDefense.jpg");
 * }
 *
 * function draw(){
 *   background(0);
 *   rotateZ(frameCount * 0.01);
 *   rotateX(frameCount * 0.01);
 *   rotateY(frameCount * 0.01);
 *   //pass image as texture
 *   texture(img);
 *   box(200, 200, 200);
 * }
 * </code>
 * </div>
 *
 * <div>
 * <code>
 * var pg;
 * function setup(){
 *   createCanvas(100, 100, WEBGL);
 *   pg = createGraphics(256,256);
 * }
 *
 * function draw(){
 *   background(0);
 *   pg.background(255);
 *   pg.text('hello world!');
 *   //pass image as texture
 *   texture(pg);
 *   plane(200);
 * }
 * </code>
 * </div>
 *
 * <div>
 * <code>
 * var vid;
 * function preload(){
 *   vid = createVideo("assets/fingers.mov");
 *   vid.hide();
 *   vid.loop();
 * }
 * function setup(){
 *   createCanvas(100, 100, WEBGL);
 * }
 *
 * function draw(){
 *   background(0);
 *   //pass video frame as texture
 *   texture(vid);
 *   plane(200);
 * }
 * </code>
 * </div>
 */
p5.prototype.texture = function(){
  var args = new Array(arguments.length);
  for (var i = 0; i < args.length; ++i) {
    args[i] = arguments[i];
  }
  var gl = this._renderer.GL;
  var shaderProgram = this._renderer._getShader('lightVert',
    'lightTextureFrag');
  gl.useProgram(shaderProgram);
  var textureData;
  //if argument is not already a texture
  //create a new one
  if(!args[0].isTexture){
    if (args[0] instanceof p5.Image) {
      textureData = args[0].canvas;
    }
    //if param is a video
    else if (typeof p5.MediaElement !== 'undefined' &&
            args[0] instanceof p5.MediaElement){
      if(!args[0].loadedmetadata) {return;}
      textureData = args[0].elt;
    }
    //used with offscreen 2d graphics renderer
    else if(args[0] instanceof p5.Graphics){
      textureData = args[0].elt;
    }
    var tex = gl.createTexture();
    args[0]._setProperty('tex', tex);
    args[0]._setProperty('isTexture', true);
    this._renderer._bind.call(this, tex, textureData);
  }
  else {
    if(args[0] instanceof p5.Graphics ||
      (typeof p5.MediaElement !== 'undefined' &&
      args[0] instanceof p5.MediaElement)){
      textureData = args[0].elt;
    }
    else if(args[0] instanceof p5.Image){
      textureData = args[0].canvas;
    }
    this._renderer._bind.call(this, args[0].tex, textureData);
  }
  //this is where we'd activate multi textures
  //eg. gl.activeTexture(gl.TEXTURE0 + (unit || 0));
  //but for now we just have a single texture.
  //@TODO need to extend this functionality
  gl.activeTexture(gl.TEXTURE0);
  gl.bindTexture(gl.TEXTURE_2D, args[0].tex);
  gl.uniform1i(gl.getUniformLocation(shaderProgram, 'isTexture'), true);
  gl.uniform1i(gl.getUniformLocation(shaderProgram, 'uSampler'), 0);
  return this;
};

/**
 * Texture Util functions
 */
p5.RendererGL.prototype._bind = function(tex, data){
  var gl = this._renderer.GL;
  gl.bindTexture(gl.TEXTURE_2D, tex);
  gl.pixelStorei(gl.UNPACK_FLIP_Y_WEBGL, true);
  gl.texImage2D(gl.TEXTURE_2D, 0,
    gl.RGBA, gl.RGBA, gl.UNSIGNED_BYTE, data);
  gl.pixelStorei(gl.UNPACK_FLIP_Y_WEBGL, true);
  gl.texParameteri(gl.TEXTURE_2D,
  gl.TEXTURE_MAG_FILTER, gl.LINEAR);
  gl.texParameteri(gl.TEXTURE_2D,
  gl.TEXTURE_MIN_FILTER, gl.LINEAR);
  gl.texParameteri(gl.TEXTURE_2D,
  gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
  gl.texParameteri(gl.TEXTURE_2D,
  gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
  gl.bindTexture(gl.TEXTURE_2D, null);
};

/**
 * Checks whether val is a pot
 * more info on power of 2 here:
 * https://www.opengl.org/wiki/NPOT_Texture
 * @param  {Number}  value
 * @return {Boolean}
 */
// function _isPowerOf2 (value){
//   return (value & (value - 1)) === 0;
// }

/**
 * returns the next highest power of 2 value
 * @param  {Number} value [description]
 * @return {Number}       [description]
 */
// function _nextHighestPOT (value){
//   --value;
//   for (var i = 1; i < 32; i <<= 1) {
//     value = value | value >> i;
//   }
//   return value + 1;

/**
 * Ambient material for geometry with a given color
 * @method  ambientMaterial
 * @param  {Number|Array|String|p5.Color} v1  gray value,
 * red or hue value (depending on the current color mode),
 * or color Array, or CSS color string
 * @param  {Number}            [v2] optional: green or saturation value
 * @param  {Number}            [v3] optional: blue or brightness value
 * @param  {Number}            [a]  optional: opacity
* @return {p5}                 the p5 object
 * @example
 * <div>
 * <code>
 * function setup(){
 *   createCanvas(100, 100, WEBGL);
 * }
 * function draw(){
 *  background(0);
 *  ambientLight(100);
 *  pointLight(250, 250, 250, 100, 100, 0);
 *  ambientMaterial(250);
 *  sphere(200);
 * }
 * </code>
 * </div>
 */
p5.prototype.ambientMaterial = function(v1, v2, v3, a) {
  var gl = this._renderer.GL;
  var shaderProgram =
    this._renderer._getShader('lightVert', 'lightTextureFrag');

  gl.useProgram(shaderProgram);
  shaderProgram.uMaterialColor = gl.getUniformLocation(
    shaderProgram, 'uMaterialColor' );
  var colors = this._renderer._applyColorBlend.apply(this._renderer, arguments);

  gl.uniform4f(shaderProgram.uMaterialColor,
    colors[0], colors[1], colors[2], colors[3]);

  shaderProgram.uSpecular = gl.getUniformLocation(
    shaderProgram, 'uSpecular' );
  gl.uniform1i(shaderProgram.uSpecular, false);

  gl.uniform1i(gl.getUniformLocation(shaderProgram, 'isTexture'), false);

  return this;
};

/**
 * Specular material for geometry with a given color
 * @method specularMaterial
 * @param  {Number|Array|String|p5.Color} v1  gray value,
 * red or hue value (depending on the current color mode),
 * or color Array, or CSS color string
 * @param  {Number}            [v2] optional: green or saturation value
 * @param  {Number}            [v3] optional: blue or brightness value
 * @param  {Number}            [a]  optional: opacity
 * @return {p5}                the p5 object
 * @example
 * <div>
 * <code>
 * function setup(){
 *   createCanvas(100, 100, WEBGL);
 * }
 * function draw(){
 *  background(0);
 *  ambientLight(100);
 *  pointLight(250, 250, 250, 100, 100, 0);
 *  specularMaterial(250);
 *  sphere(200);
 * }
 * </code>
 * </div>
 */
p5.prototype.specularMaterial = function(v1, v2, v3, a) {
  var gl = this._renderer.GL;
  var shaderProgram =
    this._renderer._getShader('lightVert', 'lightTextureFrag');
  gl.useProgram(shaderProgram);
  gl.uniform1i(gl.getUniformLocation(shaderProgram, 'isTexture'), false);
  shaderProgram.uMaterialColor = gl.getUniformLocation(
    shaderProgram, 'uMaterialColor' );
  var colors = this._renderer._applyColorBlend.apply(this._renderer, arguments);
  gl.uniform4f(shaderProgram.uMaterialColor,
    colors[0], colors[1], colors[2], colors[3]);
  shaderProgram.uSpecular = gl.getUniformLocation(
    shaderProgram, 'uSpecular' );
  gl.uniform1i(shaderProgram.uSpecular, true);

  return this;
};

/**
 * @private blends colors according to color components.
 * If alpha value is less than 1, we need to enable blending
 * on our gl context.  Otherwise opaque objects need to a depthMask.
 * @param  {Number} v1 [description]
 * @param  {Number} v2 [description]
 * @param  {Number} v3 [description]
 * @param  {Number} a  [description]
 * @return {[Number]}  Normalized numbers array
 */
p5.RendererGL.prototype._applyColorBlend = function(v1,v2,v3,a){
  var gl = this.GL;
  var color = this._pInst.color.apply(
    this._pInst, arguments);
  var colors = color._array;
  if(colors[colors.length-1] < 1.0){
    gl.depthMask(false);
    gl.enable(gl.BLEND);
    gl.blendEquation( gl.FUNC_ADD );
    gl.blendFunc( gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA );
  } else {
    gl.depthMask(true);
    gl.disable(gl.BLEND);
  }
  return colors;
};

module.exports = p5;

},{"../core/core":37}],82:[function(_dereq_,module,exports){
//some of the functions are adjusted from Three.js(http://threejs.org)

'use strict';

var p5 = _dereq_('../core/core');

/**
 * p5 Geometry class
 * @constructor
 * @param  {Function | Object} vertData callback function or Object
 *                     containing routine(s) for vertex data generation
 * @param  {Number} [detailX] number of vertices on horizontal surface
 * @param  {Number} [detailY] number of vertices on horizontal surface
 * @param {Function} [callback] function to call upon object instantiation.
 *
 */
p5.Geometry = function
(detailX, detailY, callback){
  //an array containing every vertex
  //@type [p5.Vector]
  this.vertices = [];
  //an array containing 1 normal per vertex
  //@type [p5.Vector]
  //[p5.Vector, p5.Vector, p5.Vector,p5.Vector, p5.Vector, p5.Vector,...]
  this.vertexNormals = [];
  //an array containing each three vertex indices that form a face
  //[[0, 1, 2], [2, 1, 3], ...]
  this.faces = [];
  //a 2D array containing uvs for every vertex
  //[[0.0,0.0],[1.0,0.0], ...]
  this.uvs = [];
  this.detailX = (detailX !== undefined) ? detailX: 1;
  this.detailY = (detailY !== undefined) ? detailY: 1;
  if(callback instanceof Function){
    callback.call(this);
  }
  return this;
};

p5.Geometry.prototype.computeFaces = function(){
  var sliceCount = this.detailX + 1;
  var a, b, c, d;
  for (var i = 0; i < this.detailY; i++){
    for (var j = 0; j < this.detailX; j++){
      a = i * sliceCount + j;// + offset;
      b = i * sliceCount + j + 1;// + offset;
      c = (i + 1)* sliceCount + j + 1;// + offset;
      d = (i + 1)* sliceCount + j;// + offset;
      this.faces.push([a, b, d]);
      this.faces.push([d, b, c]);
    }
  }
  return this;
};

p5.Geometry.prototype._getFaceNormal = function(faceId,vertId){
  //This assumes that vA->vB->vC is a counter-clockwise ordering
  var face = this.faces[faceId];
  var vA = this.vertices[face[vertId%3]];
  var vB = this.vertices[face[(vertId+1)%3]];
  var vC = this.vertices[face[(vertId+2)%3]];
  var n = p5.Vector.cross(
    p5.Vector.sub(vB,vA),
    p5.Vector.sub(vC,vA));
  var sinAlpha = p5.Vector.mag(n) /
  (p5.Vector.mag(p5.Vector.sub(vB,vA))*
    p5.Vector.mag(p5.Vector.sub(vC,vA)));
  n = n.normalize();
  return n.mult(Math.asin(sinAlpha));
};
/**
 * computes smooth normals per vertex as an average of each
 * face.
 */
p5.Geometry.prototype.computeNormals = function (){
  for(var v=0; v < this.vertices.length; v++){
    var normal = new p5.Vector();
    for(var i=0; i < this.faces.length; i++){
      //if our face contains a given vertex
      //calculate an average of the normals
      //of the triangles adjacent to that vertex
      if(this.faces[i][0] === v ||
        this.faces[i][1] === v ||
        this.faces[i][2] === v)
      {
        normal = normal.add(this._getFaceNormal(i, v));
      }
    }
    normal = normal.normalize();
    this.vertexNormals.push(normal);
  }
  return this;
};

/**
 * Averages the vertex normals. Used in curved
 * surfaces
 * @return {p5.Geometry}
 */
p5.Geometry.prototype.averageNormals = function() {

  for(var i = 0; i <= this.detailY; i++){
    var offset = this.detailX + 1;
    var temp = p5.Vector
      .add(this.vertexNormals[i*offset],
        this.vertexNormals[i*offset + this.detailX]);
    temp = p5.Vector.div(temp, 2);
    this.vertexNormals[i*offset] = temp;
    this.vertexNormals[i*offset + this.detailX] = temp;
  }
  return this;
};

/**
 * Averages pole normals.  Used in spherical primitives
 * @return {p5.Geometry}
 */
p5.Geometry.prototype.averagePoleNormals = function() {

  //average the north pole
  var sum = new p5.Vector(0, 0, 0);
  for(var i = 0; i < this.detailX; i++){
    sum.add(this.vertexNormals[i]);
  }
  sum = p5.Vector.div(sum, this.detailX);

  for(i = 0; i < this.detailX; i++){
    this.vertexNormals[i] = sum;
  }

  //average the south pole
  sum = new p5.Vector(0, 0, 0);
  for(i = this.vertices.length - 1;
    i > this.vertices.length - 1 - this.detailX; i--){
    sum.add(this.vertexNormals[i]);
  }
  sum = p5.Vector.div(sum, this.detailX);

  for(i = this.vertices.length - 1;
    i > this.vertices.length - 1 - this.detailX; i--){
    this.vertexNormals[i] = sum;
  }
  return this;
};

/**
 * Modifies all vertices to be centered within the range -100 to 100.
 * @return {p5.Geometry}
 */
p5.Geometry.prototype.normalize = function() {
  if(this.vertices.length > 0) {
    // Find the corners of our bounding box
    var maxPosition = this.vertices[0].copy();
    var minPosition = this.vertices[0].copy();

    for(var i = 0; i < this.vertices.length; i++) {
      maxPosition.x = Math.max(maxPosition.x, this.vertices[i].x);
      minPosition.x = Math.min(minPosition.x, this.vertices[i].x);
      maxPosition.y = Math.max(maxPosition.y, this.vertices[i].y);
      minPosition.y = Math.min(minPosition.y, this.vertices[i].y);
      maxPosition.z = Math.max(maxPosition.z, this.vertices[i].z);
      minPosition.z = Math.min(minPosition.z, this.vertices[i].z);
    }

    var center = p5.Vector.lerp(maxPosition, minPosition, 0.5);
    var dist = p5.Vector.sub(maxPosition, minPosition);
    var longestDist = Math.max(Math.max(dist.x, dist.y), dist.z);
    var scale = 200 / longestDist;

    for(i = 0; i < this.vertices.length; i++) {
      this.vertices[i].sub(center);
      this.vertices[i].mult(scale);
    }
  }
  return this;
};

module.exports = p5.Geometry;

},{"../core/core":37}],83:[function(_dereq_,module,exports){
/**
* @requires constants
* @todo see methods below needing further implementation.
* future consideration: implement SIMD optimizations
* when browser compatibility becomes available
* https://developer.mozilla.org/en-US/docs/Web/JavaScript/
*   Reference/Global_Objects/SIMD
*/

'use strict';

var p5 = _dereq_('../core/core');
var polarGeometry = _dereq_('../math/polargeometry');
var constants = _dereq_('../core/constants');
var GLMAT_ARRAY_TYPE = (
    typeof Float32Array !== 'undefined') ?
  Float32Array : Array;

/**
 * A class to describe a 4x4 matrix
 * for model and view matrix manipulation in the p5js webgl renderer.
 * class p5.Matrix
 * @constructor
 * @param {Array} [mat4] array literal of our 4x4 matrix
 */
p5.Matrix = function() {
  var args = new Array(arguments.length);
  for (var i = 0; i < args.length; ++i) {
    args[i] = arguments[i];
  }
  // This is default behavior when object
  // instantiated using createMatrix()
  // @todo implement createMatrix() in core/math.js
  if(args[0] instanceof p5) {
    // save reference to p5 if passed in
    this.p5 = args[0];
    if(args[1] === 'mat3'){
      this.mat3 = args[2] || new GLMAT_ARRAY_TYPE([
        1, 0, 0,
        0, 1, 0,
        0, 0, 1
      ]);
    }
    else {
      this.mat4  = args[1] || new GLMAT_ARRAY_TYPE([
        1, 0, 0, 0,
        0, 1, 0, 0,
        0, 0, 1, 0,
        0, 0, 0, 1
      ]);
    }
  // default behavior when object
  // instantiated using new p5.Matrix()
  } else {
    if(args[0] === 'mat3'){
      this.mat3 = args[1] || new GLMAT_ARRAY_TYPE([
        1, 0, 0,
        0, 1, 0,
        0, 0, 1
      ]);
    }
    else {
      this.mat4 = args[0] || new GLMAT_ARRAY_TYPE([
        1, 0, 0, 0,
        0, 1, 0, 0,
        0, 0, 1, 0,
        0, 0, 0, 1
      ]);
    }
  }
  return this;
};

/**
 * Sets the x, y, and z component of the vector using two or three separate
 * variables, the data from a p5.Matrix, or the values from a float array.
 *
 * @param {p5.Matrix|Array} [inMatrix] the input p5.Matrix or
 *                                     an Array of length 16
 */
p5.Matrix.prototype.set = function (inMatrix) {
  if (inMatrix instanceof p5.Matrix) {
    this.mat4 = inMatrix.mat4;
    return this;
  }
  else if (inMatrix instanceof GLMAT_ARRAY_TYPE) {
    this.mat4 = inMatrix;
    return this;
  }
  return this;
};

/**
 * Gets a copy of the vector, returns a p5.Matrix object.
 *
 * @return {p5.Matrix} the copy of the p5.Matrix object
 */
p5.Matrix.prototype.get = function () {
  return new p5.Matrix(this.mat4);
};

/**
 * return a copy of a matrix
 * @return {p5.Matrix}   the result matrix
 */
p5.Matrix.prototype.copy = function(){
  var copied = new p5.Matrix();
  copied.mat4[0] = this.mat4[0];
  copied.mat4[1] = this.mat4[1];
  copied.mat4[2] = this.mat4[2];
  copied.mat4[3] = this.mat4[3];
  copied.mat4[4] = this.mat4[4];
  copied.mat4[5] = this.mat4[5];
  copied.mat4[6] = this.mat4[6];
  copied.mat4[7] = this.mat4[7];
  copied.mat4[8] = this.mat4[8];
  copied.mat4[9] = this.mat4[9];
  copied.mat4[10] = this.mat4[10];
  copied.mat4[11] = this.mat4[11];
  copied.mat4[12] = this.mat4[12];
  copied.mat4[13] = this.mat4[13];
  copied.mat4[14] = this.mat4[14];
  copied.mat4[15] = this.mat4[15];
  return copied;
};

/**
 * return an identity matrix
 * @return {p5.Matrix}   the result matrix
 */
p5.Matrix.identity = function(){
  return new p5.Matrix();
};

/**
 * transpose according to a given matrix
 * @param  {p5.Matrix | Typed Array} a  the matrix to be based on to transpose
 * @return {p5.Matrix}                  this
 */
p5.Matrix.prototype.transpose = function(a){
  var a01, a02, a03, a12, a13, a23;
  if(a instanceof p5.Matrix){
    a01 = a.mat4[1];
    a02 = a.mat4[2];
    a03 = a.mat4[3];
    a12 = a.mat4[6];
    a13 = a.mat4[7];
    a23 = a.mat4[11];

    this.mat4[0] = a.mat4[0];
    this.mat4[1] = a.mat4[4];
    this.mat4[2] = a.mat4[8];
    this.mat4[3] = a.mat4[12];
    this.mat4[4] = a01;
    this.mat4[5] = a.mat4[5];
    this.mat4[6] = a.mat4[9];
    this.mat4[7] = a.mat4[13];
    this.mat4[8] = a02;
    this.mat4[9] = a12;
    this.mat4[10] = a.mat4[10];
    this.mat4[11] = a.mat4[14];
    this.mat4[12] = a03;
    this.mat4[13] = a13;
    this.mat4[14] = a23;
    this.mat4[15] = a.mat4[15];

  }else if(a instanceof GLMAT_ARRAY_TYPE){
    a01 = a[1];
    a02 = a[2];
    a03 = a[3];
    a12 = a[6];
    a13 = a[7];
    a23 = a[11];

    this.mat4[0] = a[0];
    this.mat4[1] = a[4];
    this.mat4[2] = a[8];
    this.mat4[3] = a[12];
    this.mat4[4] = a01;
    this.mat4[5] = a[5];
    this.mat4[6] = a[9];
    this.mat4[7] = a[13];
    this.mat4[8] = a02;
    this.mat4[9] = a12;
    this.mat4[10] = a[10];
    this.mat4[11] = a[14];
    this.mat4[12] = a03;
    this.mat4[13] = a13;
    this.mat4[14] = a23;
    this.mat4[15] = a[15];
  }
  return this;
};

/**
 * invert  matrix according to a give matrix
 * @param  {p5.Matrix or Typed Array} a   the matrix to be based on to invert
 * @return {p5.Matrix}                    this
 */
p5.Matrix.prototype.invert = function(a){
  var a00, a01, a02, a03, a10, a11, a12, a13,
  a20, a21, a22, a23, a30, a31, a32, a33;
  if(a instanceof p5.Matrix){
    a00 = a.mat4[0];
    a01 = a.mat4[1];
    a02 = a.mat4[2];
    a03 = a.mat4[3];
    a10 = a.mat4[4];
    a11 = a.mat4[5];
    a12 = a.mat4[6];
    a13 = a.mat4[7];
    a20 = a.mat4[8];
    a21 = a.mat4[9];
    a22 = a.mat4[10];
    a23 = a.mat4[11];
    a30 = a.mat4[12];
    a31 = a.mat4[13];
    a32 = a.mat4[14];
    a33 = a.mat4[15];
  }else if(a instanceof GLMAT_ARRAY_TYPE){
    a00 = a[0];
    a01 = a[1];
    a02 = a[2];
    a03 = a[3];
    a10 = a[4];
    a11 = a[5];
    a12 = a[6];
    a13 = a[7];
    a20 = a[8];
    a21 = a[9];
    a22 = a[10];
    a23 = a[11];
    a30 = a[12];
    a31 = a[13];
    a32 = a[14];
    a33 = a[15];
  }
  var b00 = a00 * a11 - a01 * a10,
  b01 = a00 * a12 - a02 * a10,
  b02 = a00 * a13 - a03 * a10,
  b03 = a01 * a12 - a02 * a11,
  b04 = a01 * a13 - a03 * a11,
  b05 = a02 * a13 - a03 * a12,
  b06 = a20 * a31 - a21 * a30,
  b07 = a20 * a32 - a22 * a30,
  b08 = a20 * a33 - a23 * a30,
  b09 = a21 * a32 - a22 * a31,
  b10 = a21 * a33 - a23 * a31,
  b11 = a22 * a33 - a23 * a32,

  // Calculate the determinant
  det = b00 * b11 - b01 * b10 + b02 * b09 + b03 * b08 -
  b04 * b07 + b05 * b06;

  if (!det) {
    return null;
  }
  det = 1.0 / det;

  this.mat4[0] = (a11 * b11 - a12 * b10 + a13 * b09) * det;
  this.mat4[1] = (a02 * b10 - a01 * b11 - a03 * b09) * det;
  this.mat4[2] = (a31 * b05 - a32 * b04 + a33 * b03) * det;
  this.mat4[3] = (a22 * b04 - a21 * b05 - a23 * b03) * det;
  this.mat4[4] = (a12 * b08 - a10 * b11 - a13 * b07) * det;
  this.mat4[5] = (a00 * b11 - a02 * b08 + a03 * b07) * det;
  this.mat4[6] = (a32 * b02 - a30 * b05 - a33 * b01) * det;
  this.mat4[7] = (a20 * b05 - a22 * b02 + a23 * b01) * det;
  this.mat4[8] = (a10 * b10 - a11 * b08 + a13 * b06) * det;
  this.mat4[9] = (a01 * b08 - a00 * b10 - a03 * b06) * det;
  this.mat4[10] = (a30 * b04 - a31 * b02 + a33 * b00) * det;
  this.mat4[11] = (a21 * b02 - a20 * b04 - a23 * b00) * det;
  this.mat4[12] = (a11 * b07 - a10 * b09 - a12 * b06) * det;
  this.mat4[13] = (a00 * b09 - a01 * b07 + a02 * b06) * det;
  this.mat4[14] = (a31 * b01 - a30 * b03 - a32 * b00) * det;
  this.mat4[15] = (a20 * b03 - a21 * b01 + a22 * b00) * det;

  return this;
};

/**
 * Inverts a 3x3 matrix
 * @return {[type]} [description]
 */
p5.Matrix.prototype.invert3x3 = function (){
  var a00 = this.mat3[0],
  a01 = this.mat3[1],
  a02 = this.mat3[2],
  a10 = this.mat3[3],
  a11 = this.mat3[4],
  a12 = this.mat3[5],
  a20 = this.mat3[6],
  a21 = this.mat3[7],
  a22 = this.mat3[8],
  b01 = a22 * a11 - a12 * a21,
  b11 = -a22 * a10 + a12 * a20,
  b21 = a21 * a10 - a11 * a20,

  // Calculate the determinant
  det = a00 * b01 + a01 * b11 + a02 * b21;
  if (!det) {
    return null;
  }
  det = 1.0 / det;
  this.mat3[0] = b01 * det;
  this.mat3[1] = (-a22 * a01 + a02 * a21) * det;
  this.mat3[2] = (a12 * a01 - a02 * a11) * det;
  this.mat3[3] = b11 * det;
  this.mat3[4] = (a22 * a00 - a02 * a20) * det;
  this.mat3[5] = (-a12 * a00 + a02 * a10) * det;
  this.mat3[6] = b21 * det;
  this.mat3[7] = (-a21 * a00 + a01 * a20) * det;
  this.mat3[8] = (a11 * a00 - a01 * a10) * det;
  return this;
};

/**
 * transposes a 3x3 p5.Matrix by a mat3
 * @param  {[Number]} mat3 1-dimensional array
 * @return {p5.Matrix} this
 */
p5.Matrix.prototype.transpose3x3 = function (mat3){
  var a01 = mat3[1], a02 = mat3[2], a12 = mat3[5];
  this.mat3[1] = mat3[3];
  this.mat3[2] = mat3[6];
  this.mat3[3] = a01;
  this.mat3[5] = mat3[7];
  this.mat3[6] = a02;
  this.mat3[7] = a12;
  return this;
};

/**
 * converts a 4x4 matrix to its 3x3 inverse tranform
 * commonly used in MVMatrix to NMatrix conversions.
 * @param  {p5.Matrix} mat4 the matrix to be based on to invert
 * @return {p5.Matrix} this with mat3
 * @todo  finish implementation
 */
p5.Matrix.prototype.inverseTranspose = function (matrix){
  if(this.mat3 === undefined){
    console.error('sorry, this function only works with mat3');
  }
  else {
    //convert mat4 -> mat3
    this.mat3[0] = matrix.mat4[0];
    this.mat3[1] = matrix.mat4[1];
    this.mat3[2] = matrix.mat4[2];
    this.mat3[3] = matrix.mat4[4];
    this.mat3[4] = matrix.mat4[5];
    this.mat3[5] = matrix.mat4[6];
    this.mat3[6] = matrix.mat4[8];
    this.mat3[7] = matrix.mat4[9];
    this.mat3[8] = matrix.mat4[10];
  }

  this.invert3x3().transpose3x3(this.mat3);
  return this;
};

/**
 * inspired by Toji's mat4 determinant
 * @return {Number} Determinant of our 4x4 matrix
 */
p5.Matrix.prototype.determinant = function(){
  var d00 = (this.mat4[0] * this.mat4[5]) - (this.mat4[1] * this.mat4[4]),
    d01 = (this.mat4[0] * this.mat4[6]) - (this.mat4[2] * this.mat4[4]),
    d02 = (this.mat4[0] * this.mat4[7]) - (this.mat4[3] * this.mat4[4]),
    d03 = (this.mat4[1] * this.mat4[6]) - (this.mat4[2] * this.mat4[5]),
    d04 = (this.mat4[1] * this.mat4[7]) - (this.mat4[3] * this.mat4[5]),
    d05 = (this.mat4[2] * this.mat4[7]) - (this.mat4[3] * this.mat4[6]),
    d06 = (this.mat4[8] * this.mat4[13]) - (this.mat4[9] * this.mat4[12]),
    d07 = (this.mat4[8] * this.mat4[14]) - (this.mat4[10] * this.mat4[12]),
    d08 = (this.mat4[8] * this.mat4[15]) - (this.mat4[11] * this.mat4[12]),
    d09 = (this.mat4[9] * this.mat4[14]) - (this.mat4[10] * this.mat4[13]),
    d10 = (this.mat4[9] * this.mat4[15]) - (this.mat4[11] * this.mat4[13]),
    d11 = (this.mat4[10] * this.mat4[15]) - (this.mat4[11] * this.mat4[14]);

  // Calculate the determinant
  return d00 * d11 - d01 * d10 + d02 * d09 +
    d03 * d08 - d04 * d07 + d05 * d06;
};

/**
 * multiply two mat4s
 * @param {p5.Matrix | Array}  multMatrix The matrix we want to multiply by
 * @return {p5.Matrix}         this
 */
p5.Matrix.prototype.mult = function(multMatrix){
  var _dest = new GLMAT_ARRAY_TYPE(16);
  var _src = new GLMAT_ARRAY_TYPE(16);

  if(multMatrix instanceof p5.Matrix) {
    _src = multMatrix.mat4;
  }
  else if(multMatrix instanceof GLMAT_ARRAY_TYPE){
    _src = multMatrix;
  }

  // each row is used for the multiplier
  var b0  = this.mat4[0], b1 = this.mat4[1],
    b2 = this.mat4[2], b3 = this.mat4[3];
  _dest[0] = b0*_src[0] + b1*_src[4] + b2*_src[8] + b3*_src[12];
  _dest[1] = b0*_src[1] + b1*_src[5] + b2*_src[9] + b3*_src[13];
  _dest[2] = b0*_src[2] + b1*_src[6] + b2*_src[10] + b3*_src[14];
  _dest[3] = b0*_src[3] + b1*_src[7] + b2*_src[11] + b3*_src[15];

  b0 = this.mat4[4];
  b1 = this.mat4[5];
  b2 = this.mat4[6];
  b3 = this.mat4[7];
  _dest[4] = b0*_src[0] + b1*_src[4] + b2*_src[8] + b3*_src[12];
  _dest[5] = b0*_src[1] + b1*_src[5] + b2*_src[9] + b3*_src[13];
  _dest[6] = b0*_src[2] + b1*_src[6] + b2*_src[10] + b3*_src[14];
  _dest[7] = b0*_src[3] + b1*_src[7] + b2*_src[11] + b3*_src[15];

  b0 = this.mat4[8];
  b1 = this.mat4[9];
  b2 = this.mat4[10];
  b3 = this.mat4[11];
  _dest[8] = b0*_src[0] + b1*_src[4] + b2*_src[8] + b3*_src[12];
  _dest[9] = b0*_src[1] + b1*_src[5] + b2*_src[9] + b3*_src[13];
  _dest[10] = b0*_src[2] + b1*_src[6] + b2*_src[10] + b3*_src[14];
  _dest[11] = b0*_src[3] + b1*_src[7] + b2*_src[11] + b3*_src[15];

  b0 = this.mat4[12];
  b1 = this.mat4[13];
  b2 = this.mat4[14];
  b3 = this.mat4[15];
  _dest[12] = b0*_src[0] + b1*_src[4] + b2*_src[8] + b3*_src[12];
  _dest[13] = b0*_src[1] + b1*_src[5] + b2*_src[9] + b3*_src[13];
  _dest[14] = b0*_src[2] + b1*_src[6] + b2*_src[10] + b3*_src[14];
  _dest[15] = b0*_src[3] + b1*_src[7] + b2*_src[11] + b3*_src[15];

  this.mat4 = _dest;

  return this;
};

/**
 * scales a p5.Matrix by scalars or a vector
 * @param  {p5.Vector | Array }
 *                      vector to scale by
 * @return {p5.Matrix}  this
 */
p5.Matrix.prototype.scale = function() {
  var x,y,z;
  var args = new Array(arguments.length);
  for (var i = 0; i < args.length; ++i) {
    args[i] = arguments[i];
  }
  //if our 1st arg is a type p5.Vector
  if (args[0] instanceof p5.Vector){
    x = args[0].x;
    y = args[0].y;
    z = args[0].z;
  }
  //otherwise if it's an array
  else if (args[0] instanceof Array){
    x = args[0][0];
    y = args[0][1];
    z = args[0][2];
  }
  var _dest = new GLMAT_ARRAY_TYPE(16);
  _dest[0] = this.mat4[0] * x;
  _dest[1] = this.mat4[1] * x;
  _dest[2] = this.mat4[2] * x;
  _dest[3] = this.mat4[3] * x;
  _dest[4] = this.mat4[4] * y;
  _dest[5] = this.mat4[5] * y;
  _dest[6] = this.mat4[6] * y;
  _dest[7] = this.mat4[7] * y;
  _dest[8] = this.mat4[8] * z;
  _dest[9] = this.mat4[9] * z;
  _dest[10] = this.mat4[10] * z;
  _dest[11] = this.mat4[11] * z;
  _dest[12] = this.mat4[12];
  _dest[13] = this.mat4[13];
  _dest[14] = this.mat4[14];
  _dest[15] = this.mat4[15];

  this.mat4 = _dest;
  return this;
};

/**
 * rotate our Matrix around an axis by the given angle.
 * @param  {Number} a The angle of rotation in radians
 * @param  {p5.Vector | Array} axis  the axis(es) to rotate around
 * @return {p5.Matrix}                    this
 * inspired by Toji's gl-matrix lib, mat4 rotation
 */
p5.Matrix.prototype.rotate = function(a, axis){
  var x, y, z, _a, len;

  if (this.p5) {
    if (this.p5._angleMode === constants.DEGREES) {
      _a = polarGeometry.degreesToRadians(a);
    }
  }
  else {
    _a = a;
  }
  if (axis instanceof p5.Vector) {
    x = axis.x;
    y = axis.y;
    z = axis.z;
  }
  else if (axis instanceof Array) {
    x = axis[0];
    y = axis[1];
    z = axis[2];
  }

  len = Math.sqrt(x * x + y * y + z * z);
  x *= (1/len);
  y *= (1/len);
  z *= (1/len);

  var a00 = this.mat4[0];
  var a01 = this.mat4[1];
  var a02 = this.mat4[2];
  var a03 = this.mat4[3];
  var a10 = this.mat4[4];
  var a11 = this.mat4[5];
  var a12 = this.mat4[6];
  var a13 = this.mat4[7];
  var a20 = this.mat4[8];
  var a21 = this.mat4[9];
  var a22 = this.mat4[10];
  var a23 = this.mat4[11];

  //sin,cos, and tan of respective angle
  var sA = Math.sin(_a);
  var cA = Math.cos(_a);
  var tA = 1 - cA;
  // Construct the elements of the rotation matrix
  var b00 = x * x * tA + cA;
  var b01 = y * x * tA + z * sA;
  var b02 = z * x * tA - y * sA;
  var b10 = x * y * tA - z * sA;
  var b11 = y * y * tA + cA;
  var b12 = z * y * tA + x * sA;
  var b20 = x * z * tA + y * sA;
  var b21 = y * z * tA - x * sA;
  var b22 = z * z * tA + cA;

  // rotation-specific matrix multiplication
  this.mat4[0] = a00 * b00 + a10 * b01 + a20 * b02;
  this.mat4[1] = a01 * b00 + a11 * b01 + a21 * b02;
  this.mat4[2] = a02 * b00 + a12 * b01 + a22 * b02;
  this.mat4[3] = a03 * b00 + a13 * b01 + a23 * b02;
  this.mat4[4] = a00 * b10 + a10 * b11 + a20 * b12;
  this.mat4[5] = a01 * b10 + a11 * b11 + a21 * b12;
  this.mat4[6] = a02 * b10 + a12 * b11 + a22 * b12;
  this.mat4[7] = a03 * b10 + a13 * b11 + a23 * b12;
  this.mat4[8] = a00 * b20 + a10 * b21 + a20 * b22;
  this.mat4[9] = a01 * b20 + a11 * b21 + a21 * b22;
  this.mat4[10] = a02 * b20 + a12 * b21 + a22 * b22;
  this.mat4[11] = a03 * b20 + a13 * b21 + a23 * b22;

  return this;
};

/**
 * @todo  finish implementing this method!
 * translates
 * @param  {Array} v vector to translate by
 * @return {p5.Matrix}                    this
 */
p5.Matrix.prototype.translate = function(v){
  var x = v[0],
    y = v[1],
    z = v[2] || 0;
  this.mat4[12] =
    this.mat4[0] * x +this.mat4[4] * y +this.mat4[8] * z +this.mat4[12];
  this.mat4[13] =
    this.mat4[1] * x +this.mat4[5] * y +this.mat4[9] * z +this.mat4[13];
  this.mat4[14] =
    this.mat4[2] * x +this.mat4[6] * y +this.mat4[10] * z +this.mat4[14];
  this.mat4[15] =
    this.mat4[3] * x +this.mat4[7] * y +this.mat4[11] * z +this.mat4[15];
};

p5.Matrix.prototype.rotateX = function(a){
  this.rotate(a, [1,0,0]);
};
p5.Matrix.prototype.rotateY = function(a){
  this.rotate(a, [0,1,0]);
};
p5.Matrix.prototype.rotateZ = function(a){
  this.rotate(a, [0,0,1]);
};

/**
 * sets the perspective matrix
 * @param  {Number} fovy   [description]
 * @param  {Number} aspect [description]
 * @param  {Number} near   near clipping plane
 * @param  {Number} far    far clipping plane
 * @return {void}
 */
p5.Matrix.prototype.perspective = function(fovy,aspect,near,far){

  var f = 1.0 / Math.tan(fovy / 2),
    nf = 1 / (near - far);

  this.mat4[0] = f / aspect;
  this.mat4[1] = 0;
  this.mat4[2] = 0;
  this.mat4[3] = 0;
  this.mat4[4] = 0;
  this.mat4[5] = f;
  this.mat4[6] = 0;
  this.mat4[7] = 0;
  this.mat4[8] = 0;
  this.mat4[9] = 0;
  this.mat4[10] = (far + near) * nf;
  this.mat4[11] = -1;
  this.mat4[12] = 0;
  this.mat4[13] = 0;
  this.mat4[14] = (2 * far * near) * nf;
  this.mat4[15] = 0;

  return this;

};

/**
 * sets the ortho matrix
 * @param  {Number} left   [description]
 * @param  {Number} right  [description]
 * @param  {Number} bottom [description]
 * @param  {Number} top    [description]
 * @param  {Number} near   near clipping plane
 * @param  {Number} far    far clipping plane
 * @return {void}
 */
p5.Matrix.prototype.ortho = function(left,right,bottom,top,near,far){

  var lr = 1 / (left - right),
    bt = 1 / (bottom - top),
    nf = 1 / (near - far);
  this.mat4[0] = -2 * lr;
  this.mat4[1] = 0;
  this.mat4[2] = 0;
  this.mat4[3] = 0;
  this.mat4[4] = 0;
  this.mat4[5] = -2 * bt;
  this.mat4[6] = 0;
  this.mat4[7] = 0;
  this.mat4[8] = 0;
  this.mat4[9] = 0;
  this.mat4[10] = 2 * nf;
  this.mat4[11] = 0;
  this.mat4[12] = (left + right) * lr;
  this.mat4[13] = (top + bottom) * bt;
  this.mat4[14] = (far + near) * nf;
  this.mat4[15] = 1;

  return this;
};

/**
 * PRIVATE
 */
// matrix methods adapted from:
// https://developer.mozilla.org/en-US/docs/Web/WebGL/
// gluPerspective
//
// function _makePerspective(fovy, aspect, znear, zfar){
//    var ymax = znear * Math.tan(fovy * Math.PI / 360.0);
//    var ymin = -ymax;
//    var xmin = ymin * aspect;
//    var xmax = ymax * aspect;
//    return _makeFrustum(xmin, xmax, ymin, ymax, znear, zfar);
//  }

////
//// glFrustum
////
//function _makeFrustum(left, right, bottom, top, znear, zfar){
//  var X = 2*znear/(right-left);
//  var Y = 2*znear/(top-bottom);
//  var A = (right+left)/(right-left);
//  var B = (top+bottom)/(top-bottom);
//  var C = -(zfar+znear)/(zfar-znear);
//  var D = -2*zfar*znear/(zfar-znear);
//  var frustrumMatrix =[
//  X, 0, A, 0,
//  0, Y, B, 0,
//  0, 0, C, D,
//  0, 0, -1, 0
//];
//return frustrumMatrix;
// }

// function _setMVPMatrices(){
////an identity matrix
////@TODO use the p5.Matrix class to abstract away our MV matrices and
///other math
//var _mvMatrix =
//[
//  1.0,0.0,0.0,0.0,
//  0.0,1.0,0.0,0.0,
//  0.0,0.0,1.0,0.0,
//  0.0,0.0,0.0,1.0
//];

module.exports = p5.Matrix;

},{"../core/constants":36,"../core/core":37,"../math/polargeometry":67}],84:[function(_dereq_,module,exports){
/**
 * Welcome to RendererGL Immediate Mode.
 * Immediate mode is used for drawing custom shapes
 * from a set of vertices.  Immediate Mode is activated
 * when you call beginShape() & de-activated when you call endShape().
 * Immediate mode is a style of programming borrowed
 * from OpenGL's (now-deprecated) immediate mode.
 * It differs from p5.js' default, Retained Mode, which caches
 * geometries and buffers on the CPU to reduce the number of webgl
 * draw calls. Retained mode is more efficient & performative,
 * however, Immediate Mode is useful for sketching quick
 * geometric ideas.
 */
'use strict';

var p5 = _dereq_('../core/core');
var constants = _dereq_('../core/constants');

/**
 * Begin shape drawing.  This is a helpful way of generating
 * custom shapes quickly.  However in WEBGL mode, application
 * performance will likely drop as a result of too many calls to
 * beginShape() / endShape().  As a high performance alternative,
 * please use p5.js geometry primitives.
 * @param  {Number} mode webgl primitives mode.  beginShape supports the
 *                       following modes:
 *                       POINTS,LINES,LINE_STRIP,LINE_LOOP,TRIANGLES,
 *                       TRIANGLE_STRIP,and TRIANGLE_FAN.
 * @return {[type]}      [description]
 */
p5.RendererGL.prototype.beginShape = function(mode){
  //default shape mode is line_strip
  this.immediateMode.shapeMode = (mode !== undefined ) ?
    mode : constants.LINE_STRIP;
  //if we haven't yet initialized our
  //immediateMode vertices & buffers, create them now!
  if(this.immediateMode.vertexPositions === undefined){
    this.immediateMode.vertexPositions = [];
    this.immediateMode.vertexColors = [];
    this.immediateMode.vertexBuffer = this.GL.createBuffer();
    this.immediateMode.colorBuffer = this.GL.createBuffer();
  } else {
    this.immediateMode.vertexPositions.length = 0;
    this.immediateMode.vertexColors.length = 0;
  }
  this.isImmediateDrawing = true;
  return this;
};
/**
 * adds a vertex to be drawn in a custom Shape.
 * @param  {Number} x x-coordinate of vertex
 * @param  {Number} y y-coordinate of vertex
 * @param  {Number} z z-coordinate of vertex
 * @return {p5.RendererGL}   [description]
 * @TODO implement handling of p5.Vector args
 */
p5.RendererGL.prototype.vertex = function(x, y, z){
  this.immediateMode.vertexPositions.push(x, y, z);
  var vertexColor = this.curFillColor || [0.5, 0.5, 0.5, 1.0];
  this.immediateMode.vertexColors.push(
    vertexColor[0],
    vertexColor[1],
    vertexColor[2],
    vertexColor[3]);
  return this;
};

/**
 * End shape drawing and render vertices to screen.
 * @return {p5.RendererGL} [description]
 */
p5.RendererGL.prototype.endShape =
function(mode, isCurve, isBezier,isQuadratic, isContour, shapeKind){
  var gl = this.GL;
  this._bindImmediateBuffers(
    this.immediateMode.vertexPositions,
    this.immediateMode.vertexColors);
  if(mode){
    if(this.drawMode === 'fill'){
      switch(this.immediateMode.shapeMode){
        case constants.LINE_STRIP:
          this.immediateMode.shapeMode = constants.TRIANGLE_FAN;
          break;
        case constants.LINES:
          this.immediateMode.shapeMode = constants.TRIANGLE_FAN;
          break;
        case constants.TRIANGLES:
          this.immediateMode.shapeMode = constants.TRIANGLE_FAN;
          break;
      }
    } else {
      switch(this.immediateMode.shapeMode){
        case constants.LINE_STRIP:
          this.immediateMode.shapeMode = constants.LINE_LOOP;
          break;
        case constants.LINES:
          this.immediateMode.shapeMode = constants.LINE_LOOP;
          break;
      }
    }
  }
  //QUADS & QUAD_STRIP are not supported primitives modes
  //in webgl.
  if(this.immediateMode.shapeMode === constants.QUADS ||
    this.immediateMode.shapeMode === constants.QUAD_STRIP){
    throw new Error('sorry, ' + this.immediateMode.shapeMode+
      ' not yet implemented in webgl mode.');
  }
  else {
    gl.enable(gl.BLEND);
    gl.drawArrays(this.immediateMode.shapeMode, 0,
      this.immediateMode.vertexPositions.length / 3);
  }
  //clear out our vertexPositions & colors arrays
  //after rendering
  this.immediateMode.vertexPositions.length = 0;
  this.immediateMode.vertexColors.length = 0;
  this.isImmediateDrawing = false;
  return this;
};
/**
 * Bind immediateMode buffers to data,
 * then draw gl arrays
 * @param  {Array} vertices Numbers array representing
 *                          vertex positions
 * @return {p5.RendererGL}
 */
p5.RendererGL.prototype._bindImmediateBuffers = function(vertices, colors){
  this._setDefaultCamera();
  var gl = this.GL;
  var shaderKey = this._getCurShaderId();
  var shaderProgram = this.mHash[shaderKey];
  //vertex position Attribute
  shaderProgram.vertexPositionAttribute =
    gl.getAttribLocation(shaderProgram, 'aPosition');
  gl.enableVertexAttribArray(shaderProgram.vertexPositionAttribute);
  gl.bindBuffer(gl.ARRAY_BUFFER, this.immediateMode.vertexBuffer);
  gl.bufferData(
    gl.ARRAY_BUFFER, new Float32Array(vertices), gl.DYNAMIC_DRAW);
  gl.vertexAttribPointer(shaderProgram.vertexPositionAttribute,
    3, gl.FLOAT, false, 0, 0);

  shaderProgram.vertexColorAttribute =
    gl.getAttribLocation(shaderProgram, 'aVertexColor');
  gl.enableVertexAttribArray(shaderProgram.vertexColorAttribute);
  gl.bindBuffer(gl.ARRAY_BUFFER, this.immediateMode.colorBuffer);
  gl.bufferData(gl.ARRAY_BUFFER,
    new Float32Array(colors),gl.DYNAMIC_DRAW);
  gl.vertexAttribPointer(shaderProgram.vertexColorAttribute,
    4, gl.FLOAT, false, 0, 0);
  //matrix
  this._setMatrixUniforms(shaderKey);
  //@todo implement in all shaders (not just immediateVert)
  //set our default point size
  // this._setUniform1f(shaderKey,
  //   'uPointSize',
  //   this.pointSize);
  return this;
};

//////////////////////////////////////////////
// COLOR
//////////////////////////////////////////////

p5.RendererGL.prototype._getColorVertexShader = function(){
  var gl = this.GL;
  var mId = 'immediateVert|vertexColorFrag';
  var shaderProgram;

  if(!this.materialInHash(mId)){
    shaderProgram =
      this._initShaders('immediateVert', 'vertexColorFrag', true);
    this.mHash[mId] = shaderProgram;
    shaderProgram.vertexColorAttribute =
    gl.getAttribLocation(shaderProgram, 'aVertexColor');
    gl.enableVertexAttribArray(shaderProgram.vertexColorAttribute);
  }else{
    shaderProgram = this.mHash[mId];
  }
  return shaderProgram;
};

module.exports = p5.RendererGL;
},{"../core/constants":36,"../core/core":37}],85:[function(_dereq_,module,exports){
//Retained Mode. The default mode for rendering 3D primitives
//in WEBGL.
'use strict';

var p5 = _dereq_('../core/core');
var hashCount = 0;
/**
 * _initBufferDefaults
 * @description initializes buffer defaults. runs each time a new geometry is
 * registered
 * @param  {String} gId  key of the geometry object
 */
p5.RendererGL.prototype._initBufferDefaults = function(gId) {
  //@TODO remove this limit on hashes in gHash
  hashCount ++;
  if(hashCount > 1000){
    var key = Object.keys(this.gHash)[0];
    delete this.gHash[key];
    hashCount --;
  }

  var gl = this.GL;
  //create a new entry in our gHash
  this.gHash[gId] = {};
  this.gHash[gId].vertexBuffer = gl.createBuffer();
  this.gHash[gId].normalBuffer = gl.createBuffer();
  this.gHash[gId].uvBuffer = gl.createBuffer();
  this.gHash[gId].indexBuffer = gl.createBuffer();
};
/**
 * createBuffers description
 * @param  {String} gId    key of the geometry object
 * @param  {p5.Geometry}  obj contains geometry data
 */
p5.RendererGL.prototype.createBuffers = function(gId, obj) {
  var gl = this.GL;
  this._setDefaultCamera();
  //initialize the gl buffers for our geom groups
  this._initBufferDefaults(gId);
  //return the current shaderProgram from our material hash
  var shaderProgram = this.mHash[this._getCurShaderId()];
  //@todo rename "numberOfItems" property to something more descriptive
  //we mult the num geom faces by 3
  this.gHash[gId].numberOfItems = obj.faces.length * 3;
  gl.bindBuffer(gl.ARRAY_BUFFER, this.gHash[gId].vertexBuffer);
  gl.bufferData(
    gl.ARRAY_BUFFER,
    new Float32Array( vToNArray(obj.vertices) ),
    gl.STATIC_DRAW);
  //vertex position
  shaderProgram.vertexPositionAttribute =
    gl.getAttribLocation(shaderProgram, 'aPosition');
  gl.enableVertexAttribArray(shaderProgram.vertexPositionAttribute);

  gl.vertexAttribPointer(
    shaderProgram.vertexPositionAttribute,
    3, gl.FLOAT, false, 0, 0);

  gl.bindBuffer(gl.ARRAY_BUFFER, this.gHash[gId].normalBuffer);
  gl.bufferData(
    gl.ARRAY_BUFFER,
    new Float32Array( vToNArray(obj.vertexNormals) ),
    gl.STATIC_DRAW);
  //vertex normal
  shaderProgram.vertexNormalAttribute =
    gl.getAttribLocation(shaderProgram, 'aNormal');
  gl.enableVertexAttribArray(shaderProgram.vertexNormalAttribute);

  gl.vertexAttribPointer(
    shaderProgram.vertexNormalAttribute,
    3, gl.FLOAT, false, 0, 0);

  gl.bindBuffer(gl.ARRAY_BUFFER, this.gHash[gId].uvBuffer);
  gl.bufferData(
    gl.ARRAY_BUFFER,
    new Float32Array( flatten(obj.uvs) ),
    gl.STATIC_DRAW);
  //texture coordinate Attribute
  shaderProgram.textureCoordAttribute =
    gl.getAttribLocation(shaderProgram, 'aTexCoord');
  gl.enableVertexAttribArray(shaderProgram.textureCoordAttribute);
  gl.vertexAttribPointer(
    shaderProgram.textureCoordAttribute,
    2, gl.FLOAT, false, 0, 0);

  gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, this.gHash[gId].indexBuffer);
  gl.bufferData(
    gl.ELEMENT_ARRAY_BUFFER,
    new Uint16Array( flatten(obj.faces) ),
    gl.STATIC_DRAW);
};

/**
 * Draws buffers given a geometry key ID
 * @param  {String} gId     ID in our geom hash
 * @return {p5.RendererGL} this
 */
p5.RendererGL.prototype.drawBuffers = function(gId) {
  this._setDefaultCamera();
  var gl = this.GL;
  var shaderKey = this._getCurShaderId();
  var shaderProgram = this.mHash[shaderKey];
  //vertex position buffer
  gl.bindBuffer(gl.ARRAY_BUFFER, this.gHash[gId].vertexBuffer);
  gl.vertexAttribPointer(
    shaderProgram.vertexPositionAttribute,
    3, gl.FLOAT, false, 0, 0);
  //normal buffer
  gl.bindBuffer(gl.ARRAY_BUFFER, this.gHash[gId].normalBuffer);
  gl.vertexAttribPointer(
    shaderProgram.vertexNormalAttribute,
    3, gl.FLOAT, false, 0, 0);
  // uv buffer
  gl.bindBuffer(gl.ARRAY_BUFFER, this.gHash[gId].uvBuffer);
  gl.vertexAttribPointer(
    shaderProgram.textureCoordAttribute,
    2, gl.FLOAT, false, 0, 0);
  //vertex index buffer
  gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, this.gHash[gId].indexBuffer);
  this._setMatrixUniforms(shaderKey);
  gl.drawElements(
    gl.TRIANGLES, this.gHash[gId].numberOfItems,
    gl.UNSIGNED_SHORT, 0);
  return this;
};
///////////////////////////////
//// UTILITY FUNCTIONS
//////////////////////////////
/**
 * turn a two dimensional array into one dimensional array
 * @param  {Array} arr 2-dimensional array
 * @return {Array}     1-dimensional array
 * [[1, 2, 3],[4, 5, 6]] -> [1, 2, 3, 4, 5, 6]
 */
function flatten(arr){
  if (arr.length>0){
    return arr.reduce(function(a, b){
      return a.concat(b);
    });
  } else {
    return [];
  }
}

/**
 * turn a p5.Vector Array into a one dimensional number array
 * @param  {Array} arr  an array of p5.Vector
 * @return {Array]}     a one dimensional array of numbers
 * [p5.Vector(1, 2, 3), p5.Vector(4, 5, 6)] ->
 * [1, 2, 3, 4, 5, 6]
 */
function vToNArray(arr){
  return flatten(arr.map(function(item){
    return [item.x, item.y, item.z];
  }));
}
module.exports = p5.RendererGL;

},{"../core/core":37}],86:[function(_dereq_,module,exports){
'use strict';

var p5 = _dereq_('../core/core');
var shader = _dereq_('./shader');
_dereq_('../core/p5.Renderer');
_dereq_('./p5.Matrix');
var uMVMatrixStack = [];
var RESOLUTION = 1000;

//@TODO should implement public method
//to override these attributes
var attributes = {
  alpha: true,
  depth: true,
  stencil: true,
  antialias: false,
  premultipliedAlpha: false,
  preserveDrawingBuffer: false
};

/**
 * @class p5.RendererGL
 * @constructor
 * @extends p5.Renderer
 * 3D graphics class.
 * @todo extend class to include public method for offscreen
 * rendering (FBO).
 *
 */
p5.RendererGL = function(elt, pInst, isMainCanvas) {
  p5.Renderer.call(this, elt, pInst, isMainCanvas);
  this._initContext();

  this.isP3D = true; //lets us know we're in 3d mode
  this.GL = this.drawingContext;
  //lights
  this.ambientLightCount = 0;
  this.directionalLightCount = 0;
  this.pointLightCount = 0;
  //camera
  this._isSetCamera = false;

  /**
   * model view, projection, & normal
   * matrices
   */
  this.uMVMatrix = new p5.Matrix();
  this.uPMatrix  = new p5.Matrix();
  this.uNMatrix = new p5.Matrix('mat3');
  //Geometry & Material hashes
  this.gHash = {};
  this.mHash = {};
  //Imediate Mode
  //default drawing is done in Retained Mode
  this.isImmediateDrawing = false;
  this.immediateMode = {};
  this.curFillColor = [0.5,0.5,0.5,1.0];
  this.curStrokeColor = [0.5,0.5,0.5,1.0];
  this.pointSize = 5.0;//default point/stroke
  return this;
};

p5.RendererGL.prototype = Object.create(p5.Renderer.prototype);

//////////////////////////////////////////////
// Setting
//////////////////////////////////////////////

p5.RendererGL.prototype._initContext = function() {
  try {
    this.drawingContext = this.canvas.getContext('webgl', attributes) ||
      this.canvas.getContext('experimental-webgl', attributes);
    if (this.drawingContext === null) {
      throw new Error('Error creating webgl context');
    } else {
      console.log('p5.RendererGL: enabled webgl context');
      var gl = this.drawingContext;
      gl.enable(gl.DEPTH_TEST);
      gl.depthFunc(gl.LEQUAL);
      gl.viewport(0, 0, gl.drawingBufferWidth, gl.drawingBufferHeight);
    }
  } catch (er) {
    throw new Error(er);
  }
};
//detect if user didn't set the camera
//then call this function below
p5.RendererGL.prototype._setDefaultCamera = function(){
  if(!this._isSetCamera){
    var _w = this.width;
    var _h = this.height;
    this.uPMatrix = p5.Matrix.identity();
    this.uPMatrix.perspective(60 / 180 * Math.PI, _w / _h, 0.1, 100);
    this._isSetCamera = true;
  }
};

p5.RendererGL.prototype._update = function() {
  this.uMVMatrix = p5.Matrix.identity();
  this.translate(0, 0, -800);
  this.ambientLightCount = 0;
  this.directionalLightCount = 0;
  this.pointLightCount = 0;
};

/**
 * [background description]
 * @return {[type]} [description]
 */
p5.RendererGL.prototype.background = function() {
  var gl = this.GL;
  var _col = this._pInst.color.apply(this._pInst, arguments);
  var _r = (_col.levels[0]) / 255;
  var _g = (_col.levels[1]) / 255;
  var _b = (_col.levels[2]) / 255;
  var _a = (_col.levels[3]) / 255;
  gl.clearColor(_r, _g, _b, _a);
  gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);
};

//@TODO implement this
// p5.RendererGL.prototype.clear = function() {
//@TODO
// };

//////////////////////////////////////////////
// SHADER
//////////////////////////////////////////////

/**
 * [_initShaders description]
 * @param  {string} vertId [description]
 * @param  {string} fragId [description]
 * @return {[type]}        [description]
 */
p5.RendererGL.prototype._initShaders =
function(vertId, fragId, isImmediateMode) {
  var gl = this.GL;
  //set up our default shaders by:
  // 1. create the shader,
  // 2. load the shader source,
  // 3. compile the shader
  var _vertShader = gl.createShader(gl.VERTEX_SHADER);
  //load in our default vertex shader
  gl.shaderSource(_vertShader, shader[vertId]);
  gl.compileShader(_vertShader);
  // if our vertex shader failed compilation?
  if (!gl.getShaderParameter(_vertShader, gl.COMPILE_STATUS)) {
    alert('Yikes! An error occurred compiling the shaders:' +
      gl.getShaderInfoLog(_vertShader));
    return null;
  }

  var _fragShader = gl.createShader(gl.FRAGMENT_SHADER);
  //load in our material frag shader
  gl.shaderSource(_fragShader, shader[fragId]);
  gl.compileShader(_fragShader);
  // if our frag shader failed compilation?
  if (!gl.getShaderParameter(_fragShader, gl.COMPILE_STATUS)) {
    alert('Darn! An error occurred compiling the shaders:' +
      gl.getShaderInfoLog(_fragShader));
    return null;
  }

  var shaderProgram = gl.createProgram();
  gl.attachShader(shaderProgram, _vertShader);
  gl.attachShader(shaderProgram, _fragShader);
  gl.linkProgram(shaderProgram);
  if (!gl.getProgramParameter(shaderProgram, gl.LINK_STATUS)) {
    alert('Snap! Error linking shader program');
  }
  //END SHADERS SETUP

  this._getLocation(shaderProgram, isImmediateMode);

  return shaderProgram;
};

p5.RendererGL.prototype._getLocation =
function(shaderProgram, isImmediateMode) {
  var gl = this.GL;
  gl.useProgram(shaderProgram);
  shaderProgram.uResolution =
    gl.getUniformLocation(shaderProgram, 'uResolution');
  gl.uniform1f(shaderProgram.uResolution, RESOLUTION);

  //projection Matrix uniform
  shaderProgram.uPMatrixUniform =
    gl.getUniformLocation(shaderProgram, 'uProjectionMatrix');
  //model view Matrix uniform
  shaderProgram.uMVMatrixUniform =
    gl.getUniformLocation(shaderProgram, 'uModelViewMatrix');

  //@TODO: figure out a better way instead of if statement
  if(isImmediateMode === undefined){
    //normal Matrix uniform
    shaderProgram.uNMatrixUniform =
    gl.getUniformLocation(shaderProgram, 'uNormalMatrix');

    shaderProgram.samplerUniform =
    gl.getUniformLocation(shaderProgram, 'uSampler');
  }
};

/**
 * Sets a shader uniform given a shaderProgram and uniform string
 * @param {String} shaderKey key to material Hash.
 * @param {String} uniform location in shader.
 * @param { Number} data data to bind uniform.  Float data type.
 * @todo currently this function sets uniform1f data.
 * Should generalize function to accept any uniform
 * data type.
 */
p5.RendererGL.prototype._setUniform1f = function(shaderKey,uniform,data)
{
  var gl = this.GL;
  var shaderProgram = this.mHash[shaderKey];
  gl.useProgram(shaderProgram);
  shaderProgram[uniform] = gl.getUniformLocation(shaderProgram, uniform);
  gl.uniform1f(shaderProgram[uniform], data);
  return this;
};

p5.RendererGL.prototype._setMatrixUniforms = function(shaderKey) {
  var gl = this.GL;
  var shaderProgram = this.mHash[shaderKey];

  gl.useProgram(shaderProgram);

  gl.uniformMatrix4fv(
    shaderProgram.uPMatrixUniform,
    false, this.uPMatrix.mat4);

  gl.uniformMatrix4fv(
    shaderProgram.uMVMatrixUniform,
    false, this.uMVMatrix.mat4);

  this.uNMatrix.inverseTranspose(this.uMVMatrix);

  gl.uniformMatrix3fv(
    shaderProgram.uNMatrixUniform,
    false, this.uNMatrix.mat3);
};
//////////////////////////////////////////////
// GET CURRENT | for shader and color
//////////////////////////////////////////////
p5.RendererGL.prototype._getShader = function(vertId, fragId, isImmediateMode) {
  var mId = vertId + '|' + fragId;
  //create it and put it into hashTable
  if(!this.materialInHash(mId)){
    var shaderProgram = this._initShaders(vertId, fragId, isImmediateMode);
    this.mHash[mId] = shaderProgram;
  }
  this.curShaderId = mId;

  return this.mHash[this.curShaderId];
};

p5.RendererGL.prototype._getCurShaderId = function(){
  //if the shader ID is not yet defined
  var mId, shaderProgram;
  if(this.drawMode !== 'fill' && this.curShaderId === undefined){
    //default shader: normalMaterial()
    mId = 'normalVert|normalFrag';
    shaderProgram = this._initShaders('normalVert', 'normalFrag');
    this.mHash[mId] = shaderProgram;
    this.curShaderId = mId;
  } else if(this.isImmediateDrawing && this.drawMode === 'fill'){
    mId = 'immediateVert|vertexColorFrag';
    shaderProgram = this._initShaders('immediateVert', 'vertexColorFrag');
    this.mHash[mId] = shaderProgram;
    this.curShaderId = mId;
  }
  return this.curShaderId;
};

//////////////////////////////////////////////
// COLOR
//////////////////////////////////////////////
/**
 * Basic fill material for geometry with a given color
 * @method  fill
 * @param  {Number|Array|String|p5.Color} v1  gray value,
 * red or hue value (depending on the current color mode),
 * or color Array, or CSS color string
 * @param  {Number}            [v2] optional: green or saturation value
 * @param  {Number}            [v3] optional: blue or brightness value
 * @param  {Number}            [a]  optional: opacity
 * @return {p5}                the p5 object
 * @example
 * <div>
 * <code>
 * function setup(){
 *   createCanvas(100, 100, WEBGL);
 * }
 *
 * function draw(){
 *  background(0);
 *  fill(250, 0, 0);
 *  rotateX(frameCount * 0.01);
 *  rotateY(frameCount * 0.01);
 *  rotateZ(frameCount * 0.01);
 *  box(200, 200, 200);
 * }
 * </code>
 * </div>
 */
p5.RendererGL.prototype.fill = function(v1, v2, v3, a) {
  var gl = this.GL;
  var shaderProgram;
  //see material.js for more info on color blending in webgl
  var colors = this._renderer._applyColorBlend.apply(this._renderer, arguments);
  this.curFillColor = colors;
  this.drawMode = 'fill';
  if(this.isImmediateDrawing){
    shaderProgram =
    this._getShader('immediateVert','vertexColorFrag');
    gl.useProgram(shaderProgram);
  } else {
    shaderProgram =
    this._getShader('normalVert', 'basicFrag');
    gl.useProgram(shaderProgram);
    //RetainedMode uses a webgl uniform to pass color vals
    //in ImmediateMode, we want access to each vertex so therefore
    //we cannot use a uniform.
    shaderProgram.uMaterialColor = gl.getUniformLocation(
      shaderProgram, 'uMaterialColor' );
    gl.uniform4f( shaderProgram.uMaterialColor,
      colors[0],
      colors[1],
      colors[2],
      colors[3]);
  }
  return this;
};
p5.RendererGL.prototype.stroke = function(r, g, b, a) {
  var color = this._pInst.color.apply(this._pInst, arguments);
  var colorNormalized = color._array;
  this.curStrokeColor = colorNormalized;
  this.drawMode = 'stroke';
  return this;
};

//@TODO
p5.RendererGL.prototype._strokeCheck = function(){
  if(this.drawMode === 'stroke'){
    throw new Error(
      'stroke for shapes in 3D not yet implemented, use fill for now :('
    );
  }
};

/**
 * [strokeWeight description]
 * @param  {Number} pointSize stroke point size
 * @return {[type]}           [description]
 * @todo  strokeWeight currently works on points only.
 * implement on all wireframes and strokes.
 */
p5.RendererGL.prototype.strokeWeight = function(pointSize) {
  this.pointSize = pointSize;
  return this;
};
//////////////////////////////////////////////
// HASH | for material and geometry
//////////////////////////////////////////////

p5.RendererGL.prototype.geometryInHash = function(gId){
  return this.gHash[gId] !== undefined;
};

p5.RendererGL.prototype.materialInHash = function(mId){
  return this.mHash[mId] !== undefined;
};

/**
 * [resize description]
 * @param  {[type]} w [description]
 * @param  {[tyoe]} h [description]
 * @return {[type]}   [description]
 */
p5.RendererGL.prototype.resize = function(w,h) {
  var gl = this.GL;
  p5.Renderer.prototype.resize.call(this, w, h);
  gl.viewport(0, 0, gl.drawingBufferWidth, gl.drawingBufferHeight);
};

/**
 * clears color and depth buffers
 * with r,g,b,a
 * @param {Number} r normalized red val.
 * @param {Number} g normalized green val.
 * @param {Number} b normalized blue val.
 * @param {Number} a normalized alpha val.
 */
p5.RendererGL.prototype.clear = function() {
  var gl = this.GL;
  gl.clearColor(arguments[0],
    arguments[1],
    arguments[2],
    arguments[3]);
  gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);
};

/**
 * [translate description]
 * @param  {[type]} x [description]
 * @param  {[type]} y [description]
 * @param  {[type]} z [description]
 * @return {[type]}   [description]
 * @todo implement handle for components or vector as args
 */
p5.RendererGL.prototype.translate = function(x, y, z) {
  //@TODO: figure out how to fit the resolution
  x = x / RESOLUTION;
  y = -y / RESOLUTION;
  z = z / RESOLUTION;
  this.uMVMatrix.translate([x,y,z]);
  return this;
};

/**
 * Scales the Model View Matrix by a vector
 * @param  {Number | p5.Vector | Array} x [description]
 * @param  {Number} [y] y-axis scalar
 * @param  {Number} [z] z-axis scalar
 * @return {this}   [description]
 */
p5.RendererGL.prototype.scale = function(x,y,z) {
  this.uMVMatrix.scale([x,y,z]);
  return this;
};

p5.RendererGL.prototype.rotate = function(rad, axis){
  this.uMVMatrix.rotate(rad, axis);
  return this;
};

p5.RendererGL.prototype.rotateX = function(rad) {
  this.rotate(rad, [1,0,0]);
  return this;
};

p5.RendererGL.prototype.rotateY = function(rad) {
  this.rotate(rad, [0,1,0]);
  return this;
};

p5.RendererGL.prototype.rotateZ = function(rad) {
  this.rotate(rad, [0,0,1]);
  return this;
};

/**
 * pushes a copy of the model view matrix onto the
 * MV Matrix stack.
 */
p5.RendererGL.prototype.push = function() {
  uMVMatrixStack.push(this.uMVMatrix.copy());
};

/**
 * [pop description]
 * @return {[type]} [description]
 */
p5.RendererGL.prototype.pop = function() {
  if (uMVMatrixStack.length === 0) {
    throw new Error('Invalid popMatrix!');
  }
  this.uMVMatrix = uMVMatrixStack.pop();
};

p5.RendererGL.prototype.resetMatrix = function() {
  this.uMVMatrix = p5.Matrix.identity();
  this.translate(0, 0, -800);
  return this;
};

// Text/Typography
// @TODO:
p5.RendererGL.prototype._applyTextProperties = function() {
  //@TODO finish implementation
  console.error('text commands not yet implemented in webgl');
};
module.exports = p5.RendererGL;

},{"../core/core":37,"../core/p5.Renderer":43,"./p5.Matrix":83,"./shader":88}],87:[function(_dereq_,module,exports){
/**
 * @module Shape
 * @submodule 3D Primitives
 * @for p5
 * @requires core
 * @requires p5.Geometry
 */

'use strict';

var p5 = _dereq_('../core/core');
_dereq_('./p5.Geometry');
/**
 * Draw a plane with given a width and height
 * @method plane
 * @param  {Number} width      width of the plane
 * @param  {Number} height     height of the plane
 * @param  {Number} [detailX]  Optional number of triangle
 *                             subdivisions in x-dimension
 * @param {Number} [detailY]   Optional number of triangle
 *                             subdivisions in y-dimension
 * @return {p5}                the p5 object
 * @example
 * <div>
 * <code>
 * //draw a plane with width 200 and height 200
 * function setup(){
 *   createCanvas(100, 100, WEBGL);
 * }
 *
 * function draw(){
 *   background(200);
 *   plane(200, 200);
 * }
 * </code>
 * </div>
 */
p5.prototype.plane = function(){
  var args = new Array(arguments.length);
  for (var i = 0; i < args.length; ++i) {
    args[i] = arguments[i];
  }
  var width = args[0] || 50;
  var height = args[1] || width;
  var detailX = typeof args[2] === 'number' ? args[2] : 1;
  var detailY = typeof args[3] === 'number' ? args[3] : 1;

  var gId = 'plane|'+width+'|'+height+'|'+detailX+'|'+detailY;

  if(!this._renderer.geometryInHash(gId)){
    var _plane = function(){
      var u,v,p;
      for (var i = 0; i <= this.detailY; i++){
        v = i / this.detailY;
        for (var j = 0; j <= this.detailX; j++){
          u = j / this.detailX;
          p = new p5.Vector(width * u - width/2,
            height * v - height/2,
            0);
          this.vertices.push(p);
          this.uvs.push([u,v]);
        }
      }
    };
    var planeGeom =
    new p5.Geometry(detailX, detailY, _plane);
    planeGeom
      .computeFaces()
      .computeNormals();
    this._renderer.createBuffers(gId, planeGeom);
  }

  this._renderer.drawBuffers(gId);

};

/**
 * Draw a box with given width, height and depth
 * @method  box
 * @param  {Number} width     width of the box
 * @param  {Number} Height    height of the box
 * @param  {Number} depth     depth of the box
 * @param {Number} [detailX]  Optional number of triangle
 *                            subdivisions in x-dimension
 * @param {Number} [detailY]  Optional number of triangle
 *                            subdivisions in y-dimension
 * @return {p5}               the p5 object
 * @example
 * <div>
 * <code>
 * //draw a spining box with width, height and depth 200
 * function setup(){
 *   createCanvas(100, 100, WEBGL);
 * }
 *
 * function draw(){
 *   background(200);
 *   rotateX(frameCount * 0.01);
 *   rotateY(frameCount * 0.01);
 *   box(200, 200, 200);
 * }
 * </code>
 * </div>
 */
p5.prototype.box = function(){
  var args = new Array(arguments.length);
  for (var i = 0; i < args.length; ++i) {
    args[i] = arguments[i];
  }
  var width = args[0] || 50;
  var height = args[1] || width;
  var depth = args[2] || width;

  var detailX = typeof args[3] === 'number' ? args[3] : 4;
  var detailY = typeof args[4] === 'number' ? args[4] : 4;
  var gId = 'box|'+width+'|'+height+'|'+depth+'|'+detailX+'|'+detailY;

  if(!this._renderer.geometryInHash(gId)){
    var _box = function(){
      var cubeIndices = [
        [0, 4, 2, 6],// -1, 0, 0],// -x
        [1, 3, 5, 7],// +1, 0, 0],// +x
        [0, 1, 4, 5],// 0, -1, 0],// -y
        [2, 6, 3, 7],// 0, +1, 0],// +y
        [0, 2, 1, 3],// 0, 0, -1],// -z
        [4, 5, 6, 7]// 0, 0, +1] // +z
      ];
      var id=0;
      for (var i = 0; i < cubeIndices.length; i++) {
        var cubeIndex = cubeIndices[i];
        var v = i * 4;
        for (var j = 0; j < 4; j++) {
          var d = cubeIndex[j];
          //inspired by lightgl:
          //https://github.com/evanw/lightgl.js
          //octants:https://en.wikipedia.org/wiki/Octant_(solid_geometry)
          var octant = new p5.Vector(
            ((d & 1) * 2 - 1)*width/2,
            ((d & 2) - 1) *height/2,
            ((d & 4) / 2 - 1) * depth/2);
          this.vertices.push( octant );
          this.uvs.push([j & 1, (j & 2) / 2]);
          id++;
        }
        this.faces.push([v, v + 1, v + 2]);
        this.faces.push([v + 2, v + 1, v + 3]);
      }
    };
    var boxGeom = new p5.Geometry(detailX,detailY, _box);
    boxGeom.computeNormals();
    //initialize our geometry buffer with
    //the key val pair:
    //geometry Id, Geom object
    this._renderer.createBuffers(gId, boxGeom);
  }
  this._renderer.drawBuffers(gId);

  return this;

};

/**
 * Draw a sphere with given radius
 * @method sphere
 * @param  {Number} radius            radius of circle
 * @param  {Number} [detailX]         optional: number of segments,
 *                                    the more segments the smoother geometry
 *                                    default is 24
 * @param  {Number} [detailY]         optional: number of segments,
 *                                    the more segments the smoother geometry
 *                                    default is 16
 * @return {p5}                       the p5 object
 * @example
 * <div>
 * <code>
 * // draw a sphere with radius 200
 * function setup(){
 *   createCanvas(100, 100, WEBGL);
 * }
 *
 * function draw(){
 *   background(200);
 *   sphere(200);
 * }
 * </code>
 * </div>
 */
p5.prototype.sphere = function(){
  var args = new Array(arguments.length);
  for (var i = 0; i < args.length; ++i) {
    args[i] = arguments[i];
  }
  //@todo validate params here
  //
  var radius = args[0] || 50;
  var detailX = typeof args[1] === 'number' ? args[1] : 24;
  var detailY = typeof args[2] === 'number' ? args[2] : 16;
  var gId = 'sphere|'+radius+'|'+detailX+'|'+detailY;
  if(!this._renderer.geometryInHash(gId)){
    var _sphere = function(){
      var u,v,p;
      for (var i = 0; i <= this.detailY; i++){
        v = i / this.detailY;
        for (var j = 0; j <= this.detailX; j++){
          u = j / this.detailX;
          var theta = 2 * Math.PI * u;
          var phi = Math.PI * v - Math.PI / 2;
          p = new p5.Vector(radius * Math.cos(phi) * Math.sin(theta),
            radius * Math.sin(phi),
            radius * Math.cos(phi) * Math.cos(theta));
          this.vertices.push(p);
          this.uvs.push([u,v]);
        }
      }
    };
    var sphereGeom = new p5.Geometry(detailX, detailY, _sphere);
    sphereGeom
      .computeFaces()
      .computeNormals()
      .averageNormals()
      .averagePoleNormals();
    this._renderer.createBuffers(gId, sphereGeom);
  }
  this._renderer.drawBuffers(gId);

  return this;
};


/**
* @private
* helper function for creating both cones and cyllinders
*/
var _truncatedCone = function(
  bottomRadius,
  topRadius,
  height,
  detailX,
  detailY,
  topCap,
  bottomCap) {
  detailX = (detailX < 3) ? 3 : detailX;
  detailY = (detailY < 1) ? 1 : detailY;
  topCap = (topCap === undefined) ? true : topCap;
  bottomCap = (bottomCap === undefined) ? true : bottomCap;
  var extra = (topCap ? 2 : 0) + (bottomCap ? 2 : 0);
  var vertsAroundEdge = detailX + 1;

  // ensure constant slant
  var slant = Math.atan2(bottomRadius - topRadius, height);
  var start = topCap ? -2 : 0;
  var end = detailY + (bottomCap ? 2 : 0);
  var yy, ii;
  for (yy = start; yy <= end; ++yy) {
    var v = yy / detailY;
    var y = height * v;
    var ringRadius;
    if (yy < 0) {
      y = 0;
      v = 1;
      ringRadius = bottomRadius;
    } else if (yy > detailY) {
      y = height;
      v = 1;
      ringRadius = topRadius;
    } else {
      ringRadius = bottomRadius +
        (topRadius - bottomRadius) * (yy / detailY);
    }
    if (yy === -2 || yy === detailY + 2) {
      ringRadius = 0;
      v = 0;
    }
    y -= height / 2;
    for (ii = 0; ii < vertsAroundEdge; ++ii) {
      //VERTICES
      this.vertices.push(
        new p5.Vector(
          Math.sin(ii*Math.PI * 2 /detailX) * ringRadius,
          y,
          Math.cos(ii*Math.PI * 2 /detailX) * ringRadius)
        );
      //VERTEX NORMALS
      this.vertexNormals.push(
        new p5.Vector(
          (yy < 0 || yy > detailY) ? 0 :
          (Math.sin(ii * Math.PI * 2 / detailX) * Math.cos(slant)),
          (yy < 0) ? -1 : (yy > detailY ? 1 : Math.sin(slant)),
          (yy < 0 || yy > detailY) ? 0 :
          (Math.cos(ii * Math.PI * 2 / detailX) * Math.cos(slant)))
        );
      //UVs
      this.uvs.push([(ii / detailX), v]);
    }
  }
  for (yy = 0; yy < detailY + extra; ++yy) {
    for (ii = 0; ii < detailX; ++ii) {
      this.faces.push([vertsAroundEdge * (yy + 0) + 0 + ii,
        vertsAroundEdge * (yy + 0) + 1 + ii,
        vertsAroundEdge * (yy + 1) + 1 + ii]);
      this.faces.push([vertsAroundEdge * (yy + 0) + 0 + ii,
        vertsAroundEdge * (yy + 1) + 1 + ii,
        vertsAroundEdge * (yy + 1) + 0 + ii]);
    }
  }
};

/**
 * Draw a cylinder with given radius and height
 * @method  cylinder
 * @param  {Number} radius     radius of the surface
 * @param  {Number} height     height of the cylinder
 * @param  {Number} [detailX]  optional: number of segments,
 *                             the more segments the smoother geometry
 *                             default is 24
 * @param {Number} [detailY]   optional: number of segments in y-dimension,
 *                             the more segments the smoother geometry
 *                             default is 16
 * @return {p5}                the p5 object
 * @example
 * <div>
 * <code>
 * //draw a spining sylinder with radius 200 and height 200
 * function setup(){
 *   createCanvas(100, 100, WEBGL);
 * }
 *
 * function draw(){
 *   background(200);
 *   rotateX(frameCount * 0.01);
 *   rotateZ(frameCount * 0.01);
 *   cylinder(200, 200);
 * }
 * </code>
 * </div>
 */
p5.prototype.cylinder = function(){
  var args = new Array(arguments.length);
  for (var i = 0; i < args.length; ++i) {
    args[i] = arguments[i];
  }
  var radius = args[0] || 50;
  var height = args[1] || radius;
  var detailX = typeof args[2] === 'number' ? args[2] : 24;
  var detailY = typeof args[3] === 'number' ? args[3] : 16;
  var gId = 'cylinder|'+radius+'|'+height+'|'+detailX+'|'+detailY;
  if(!this._renderer.geometryInHash(gId)){
    var cylinderGeom = new p5.Geometry(detailX, detailY);
    _truncatedCone.call(
      cylinderGeom,
      radius,
      radius,
      height,
      detailX,
      detailY,
      true,true);
    cylinderGeom.computeNormals();
    this._renderer.createBuffers(gId, cylinderGeom);
  }

  this._renderer.drawBuffers(gId);

  return this;
};


/**
 * Draw a cone with given radius and height
 * @method cone
 * @param  {Number} radius            radius of the bottom surface
 * @param  {Number} height            height of the cone
 * @param  {Number} [detailX]         optional: number of segments,
 *                                    the more segments the smoother geometry
 *                                    default is 24
 * @param  {Number} [detailY]         optional: number of segments,
 *                                    the more segments the smoother geometry
 *                                    default is 16
 * @return {p5}                       the p5 object
 * @example
 * <div>
 * <code>
 * //draw a spining cone with radius 200 and height 200
 * function setup(){
 *   createCanvas(100, 100, WEBGL);
 * }
 *
 * function draw(){
 *   background(200);
 *   rotateX(frameCount * 0.01);
 *   rotateZ(frameCount * 0.01);
 *   cone(200, 200);
 * }
 * </code>
 * </div>
 */
p5.prototype.cone = function(){
  var args = new Array(arguments.length);
  for (var i = 0; i < args.length; ++i) {
    args[i] = arguments[i];
  }
  var baseRadius = args[0] || 50;
  var height = args[1] || baseRadius;
  var detailX = typeof args[2] === 'number' ? args[2] : 24;
  var detailY = typeof args[3] === 'number' ? args[3] : 16;
  var gId = 'cone|'+baseRadius+'|'+height+'|'+detailX+'|'+detailY;
  if(!this._renderer.geometryInHash(gId)){
    var coneGeom = new p5.Geometry(detailX, detailY);
    _truncatedCone.call(coneGeom,
      baseRadius,
      0,//top radius 0
      height,
      detailX,
      detailY,
      true,
      true);
    //for cones we need to average Normals
    coneGeom
      .computeNormals();
    this._renderer.createBuffers(gId, coneGeom);
  }

  this._renderer.drawBuffers(gId);

  return this;
};

/**
 * Draw an ellipsoid with given raduis
 * @method ellipsoid
 * @param  {Number} radiusx           xradius of circle
 * @param  {Number} radiusy           yradius of circle
 * @param  {Number} radiusz           zradius of circle
 * @param  {Number} [detailX]         optional: number of segments,
 *                                    the more segments the smoother geometry
 *                                    default is 24. Avoid detail number above
 *                                    150, it may crash the browser.
 * @param  {Number} [detailY]         optional: number of segments,
 *                                    the more segments the smoother geometry
 *                                    default is 16. Avoid detail number above
 *                                    150, it may crash the browser.
 * @return {p5}                       the p5 object
 * @example
 * <div>
 * <code>
 * // draw an ellipsoid with radius 200, 300 and 400 .
 * function setup(){
 *   createCanvas(100, 100, WEBGL);
 * }
 *
 * function draw(){
 *   background(200);
 *   ellipsoid(200,300,400);
 * }
 * </code>
 * </div>
 */
p5.prototype.ellipsoid = function(){
  var args = new Array(arguments.length);
  for (var i = 0; i < args.length; ++i) {
    args[i] = arguments[i];
  }
  var detailX = typeof args[3] === 'number' ? args[3] : 24;
  var detailY = typeof args[4] === 'number' ? args[4] : 24;
  var radiusX = args[0] || 50;
  var radiusY = args[1] || radiusX;
  var radiusZ = args[2] || radiusX;

  var gId = 'ellipsoid|'+radiusX+'|'+radiusY+
  '|'+radiusZ+'|'+detailX+'|'+detailY;


  if(!this._renderer.geometryInHash(gId)){
    var _ellipsoid = function(){
      var u,v,p;
      for (var i = 0; i <= this.detailY; i++){
        v = i / this.detailY;
        for (var j = 0; j <= this.detailX; j++){
          u = j / this.detailX;
          var theta = 2 * Math.PI * u;
          var phi = Math.PI * v - Math.PI / 2;
          p = new p5.Vector(radiusX * Math.cos(phi) * Math.sin(theta),
            radiusY * Math.sin(phi),
            radiusZ * Math.cos(phi) * Math.cos(theta));
          this.vertices.push(p);
          this.uvs.push([u,v]);
        }
      }
    };
    var ellipsoidGeom = new p5.Geometry(detailX, detailY,_ellipsoid);
    ellipsoidGeom
      .computeFaces()
      .computeNormals();
    this._renderer.createBuffers(gId, ellipsoidGeom);
  }

  this._renderer.drawBuffers(gId);

  return this;
};

/**
 * Draw a torus with given radius and tube radius
 * @method torus
 * @param  {Number} radius        radius of the whole ring
 * @param  {Number} tubeRadius    radius of the tube
 * @param  {Number} [detailX]     optional: number of segments in x-dimension,
 *                                the more segments the smoother geometry
 *                                default is 24
 * @param  {Number} [detailY]     optional: number of segments in y-dimension,
 *                                the more segments the smoother geometry
 *                                default is 16
 * @return {p5}                   the p5 object
 * @example
 * <div>
 * <code>
 * //draw a spining torus with radius 200 and tube radius 60
 * function setup(){
 *   createCanvas(100, 100, WEBGL);
 * }
 *
 * function draw(){
 *   background(200);
 *   rotateX(frameCount * 0.01);
 *   rotateY(frameCount * 0.01);
 *   torus(200, 60);
 * }
 * </code>
 * </div>
 */
p5.prototype.torus = function(){
  var args = new Array(arguments.length);
  for (var i = 0; i < args.length; ++i) {
    args[i] = arguments[i];
  }
  var detailX = typeof args[2] === 'number' ? args[2] : 24;
  var detailY = typeof args[3] === 'number' ? args[3] : 16;

  var radius = args[0] || 50;
  var tubeRadius = args[1] || 10;

  var gId = 'torus|'+radius+'|'+tubeRadius+'|'+detailX+'|'+detailY;

  if(!this._renderer.geometryInHash(gId)){
    var _torus = function(){
      var u,v,p;
      for (var i = 0; i <= this.detailY; i++){
        v = i / this.detailY;
        for (var j = 0; j <= this.detailX; j++){
          u = j / this.detailX;
          var theta = 2 * Math.PI * u;
          var phi = 2 * Math.PI * v;
          p = new p5.Vector(
            (radius + tubeRadius * Math.cos(phi)) * Math.cos(theta),
            (radius + tubeRadius * Math.cos(phi)) * Math.sin(theta),
            tubeRadius * Math.sin(phi));
          this.vertices.push(p);
          this.uvs.push([u,v]);
        }
      }
    };
    var torusGeom = new p5.Geometry(detailX, detailY, _torus);
    torusGeom
      .computeFaces()
      .computeNormals()
      .averageNormals();
    this._renderer.createBuffers(gId, torusGeom);
  }

  this._renderer.drawBuffers(gId);

  return this;
};

///////////////////////
/// 2D primitives
/////////////////////////

//@TODO
p5.RendererGL.prototype.point = function(x, y, z){
  console.log('point not yet implemented in webgl');
  return this;
};

p5.RendererGL.prototype.triangle = function
(args){
  var x1=args[0], y1=args[1], z1=args[2];
  var x2=args[3], y2=args[4], z2=args[5];
  var x3=args[6], y3=args[7], z3=args[8];
  var gId = 'tri|'+x1+'|'+y1+'|'+z1+'|'+
  x2+'|'+y2+'|'+z2+
  x3+'|'+y3+'|'+z3;
  if(!this.geometryInHash(gId)){
    var _triangle = function(){
      var vertices = [];
      vertices.push(new p5.Vector(x1,y1,z1));
      vertices.push(new p5.Vector(x2,y2,z2));
      vertices.push(new p5.Vector(x3,y3,z3));
      this.vertices = vertices;
      this.faces = [[0,1,2]];
      this.uvs = [[0,0],[0,1],[1,1]];
    };
    var triGeom = new p5.Geometry(1,1,_triangle);
    triGeom.computeNormals();
    this.createBuffers(gId, triGeom);
  }

  this.drawBuffers(gId);
  return this;
};

p5.RendererGL.prototype.ellipse = function
(args){
  var x = args[0];
  var y = args[1];
  var z = args[2];
  var width = args[3];
  var height = args[4];
  //detailX and Y are optional 6th & 7th
  //arguments
  var detailX = args[5] || 24;
  var detailY = args[6] || 16;
  var gId = 'ellipse|'+args[0]+'|'+args[1]+'|'+args[2]+'|'+
  args[3]+'|'+args[4];
  if(!this.geometryInHash(gId)){
    var _ellipse = function(){
      var u,v,p;
      for (var i = 0; i <= this.detailY; i++){
        v = i / this.detailY;
        for (var j = 0; j <= this.detailX; j++){
          u = j / this.detailX;
          var theta = 2 * Math.PI * u;
          if(v === 0){
            p = new p5.Vector(x, y, z);
          }
          else{
            var _x = x + width * Math.sin(theta);
            var _y = y + height * Math.cos(theta);
            var _z = z;
            p = new p5.Vector(_x, _y, _z);
          }
          this.vertices.push(p);
          this.uvs.push([u,v]);
        }
      }
    };
    var ellipseGeom = new p5.Geometry(detailX,detailY,_ellipse);
    ellipseGeom
      .computeFaces()
      .computeNormals();
    this.createBuffers(gId, ellipseGeom);
  }
  this.drawBuffers(gId);
  return this;
};

p5.RendererGL.prototype.rect = function
(args){
  var gId = 'rect|'+args[0]+'|'+args[1]+'|'+args[2]+'|'+
  args[3]+'|'+args[4];
  var x = args[0];
  var y = args[1];
  var z = args[2];
  var width = args[3];
  var height = args[4];
  var detailX = args[5] || 24;
  var detailY = args[6] || 16;
  if(!this.geometryInHash(gId)){
    var _rect = function(){
      var u,v,p;
      for (var i = 0; i <= this.detailY; i++){
        v = i / this.detailY;
        for (var j = 0; j <= this.detailX; j++){
          u = j / this.detailX;
          p = new p5.Vector(
            (x + width) * u,
            (y + height) * v,
            z
          );
          this.vertices.push(p);
          this.uvs.push([u,v]);
        }
      }
    };
    var rectGeom = new p5.Geometry(detailX,detailY,_rect);
    rectGeom
      .computeFaces()
      .computeNormals();
    this.createBuffers(gId, rectGeom);
  }
  this.drawBuffers(gId);
  return this;
};

p5.RendererGL.prototype.quad = function(){
  var args = new Array(arguments.length);
  for (var i = 0; i < args.length; ++i) {
    args[i] = arguments[i];
  }
  //@todo validate params here
  //
  var x1 = args[0],
    y1 = args[1],
    z1 = args[2],
    x2 = args[3],
    y2 = args[4],
    z2 = args[5],
    x3 = args[6],
    y3 = args[7],
    z3 = args[8],
    x4 = args[9],
    y4 = args[10],
    z4 = args[11];
  var gId = 'quad|'+x1+'|'+y1+'|'+z1+'|'+
  x2+'|'+y2+'|'+z2+
  x3+'|'+y3+'|'+z3+
  x4+'|'+y4+'|'+z4;
  if(!this.geometryInHash(gId)){
    var _quad = function(){
      this.vertices.push(new p5.Vector(x1,y1,z1));
      this.vertices.push(new p5.Vector(x2,y2,z2));
      this.vertices.push(new p5.Vector(x3,y3,z3));
      this.vertices.push(new p5.Vector(x4,y4,z4));
      this.uvs.push([0, 0], [1, 0], [1, 1], [0, 1]);
    };
    var quadGeom = new p5.Geometry(2,2,_quad);
    quadGeom.computeNormals();
    quadGeom.faces = [[0,1,2],[2,3,0]];
    this.createBuffers(gId, quadGeom);
  }
  this.drawBuffers(gId);
  return this;
};

//this implementation of bezier curve
//is based on Bernstein polynomial
p5.RendererGL.prototype.bezier = function
(args){
  var bezierDetail=args[12] || 20;//value of Bezier detail
  this.beginShape();
  var coeff=[0,0,0,0];//  Bernstein polynomial coeffecients
  var vertex=[0,0,0]; //(x,y,z) coordinates of points in bezier curve
  for(var i=0; i<=bezierDetail; i++){
    coeff[0]=Math.pow(1-(i/bezierDetail),3);
    coeff[1]=(3*(i/bezierDetail)) * (Math.pow(1-(i/bezierDetail),2));
    coeff[2]=(3*Math.pow(i/bezierDetail,2)) * (1-(i/bezierDetail));
    coeff[3]=Math.pow(i/bezierDetail,3);
    vertex[0]=args[0]*coeff[0] + args[3]*coeff[1] +
              args[6]*coeff[2] + args[9]*coeff[3];
    vertex[1]=args[1]*coeff[0] + args[4]*coeff[1] +
              args[7]*coeff[2] + args[10]*coeff[3];
    vertex[2]=args[2]*coeff[0] + args[5]*coeff[1] +
              args[8]*coeff[2] + args[11]*coeff[3];
    this.vertex(vertex[0],vertex[1],vertex[2]);
  }
  this.endShape();
  return this;
};

p5.RendererGL.prototype.curve=function
(args){
  var curveDetail=args[12];
  this.beginShape();
  var coeff=[0,0,0,0];//coeffecients of the equation
  var vertex=[0,0,0]; //(x,y,z) coordinates of points in bezier curve
  for(var i=0; i<=curveDetail; i++){
    coeff[0]=Math.pow((i/curveDetail),3) * 0.5;
    coeff[1]=Math.pow((i/curveDetail),2) * 0.5;
    coeff[2]=(i/curveDetail) * 0.5;
    coeff[3]=0.5;
    vertex[0]=coeff[0]*(-args[0] + (3*args[3]) - (3*args[6]) +args[9]) +
              coeff[1]*((2*args[0]) - (5*args[3]) + (4*args[6]) - args[9]) +
              coeff[2]*(-args[0] + args[6]) +
              coeff[3]*(2*args[3]);
    vertex[1]=coeff[0]*(-args[1] + (3*args[4]) - (3*args[7]) +args[10]) +
              coeff[1]*((2*args[1]) - (5*args[4]) + (4*args[7]) - args[10]) +
              coeff[2]*(-args[1] + args[7]) +
              coeff[3]*(2*args[4]);
    vertex[2]=coeff[0]*(-args[2] + (3*args[5]) - (3*args[8]) +args[11]) +
              coeff[1]*((2*args[2]) - (5*args[5]) + (4*args[8]) - args[11]) +
              coeff[2]*(-args[2] + args[8]) +
              coeff[3]*(2*args[5]);
    this.vertex(vertex[0],vertex[1],vertex[2]);
  }
  this.endShape();
  return this;
};

module.exports = p5;

},{"../core/core":37,"./p5.Geometry":82}],88:[function(_dereq_,module,exports){


module.exports = {
  immediateVert:
    "attribute vec3 aPosition;\nattribute vec4 aVertexColor;\n\nuniform mat4 uModelViewMatrix;\nuniform mat4 uProjectionMatrix;\nuniform float uResolution;\nuniform float uPointSize;\n\nvarying vec4 vColor;\nvoid main(void) {\n  vec4 positionVec4 = vec4(aPosition / uResolution *vec3(1.0, -1.0, 1.0), 1.0);\n  gl_Position = uProjectionMatrix * uModelViewMatrix * positionVec4;\n  vColor = aVertexColor;\n  gl_PointSize = uPointSize;\n}\n",
  vertexColorVert:
    "attribute vec3 aPosition;\nattribute vec4 aVertexColor;\n\nuniform mat4 uModelViewMatrix;\nuniform mat4 uProjectionMatrix;\nuniform float uResolution;\n\nvarying vec4 vColor;\n\nvoid main(void) {\n  vec4 positionVec4 = vec4(aPosition / uResolution * vec3(1.0, -1.0, 1.0), 1.0);\n  gl_Position = uProjectionMatrix * uModelViewMatrix * positionVec4;\n  vColor = aVertexColor;\n}\n",
  vertexColorFrag:
    "precision mediump float;\nvarying vec4 vColor;\nvoid main(void) {\n  gl_FragColor = vColor;\n}",
  normalVert:
    "attribute vec3 aPosition;\nattribute vec3 aNormal;\nattribute vec2 aTexCoord;\n\nuniform mat4 uModelViewMatrix;\nuniform mat4 uProjectionMatrix;\nuniform mat3 uNormalMatrix;\nuniform float uResolution;\n\nvarying vec3 vVertexNormal;\nvarying highp vec2 vVertTexCoord;\n\nvoid main(void) {\n  vec4 positionVec4 = vec4(aPosition / uResolution, 1.0);\n  gl_Position = uProjectionMatrix * uModelViewMatrix * positionVec4;\n  vVertexNormal = vec3( uNormalMatrix * aNormal );\n  vVertTexCoord = aTexCoord;\n}\n",
  normalFrag:
    "precision mediump float;\nvarying vec3 vVertexNormal;\nvoid main(void) {\n  gl_FragColor = vec4(vVertexNormal, 1.0);\n}",
  basicFrag:
    "precision mediump float;\nvarying vec3 vVertexNormal;\nuniform vec4 uMaterialColor;\nvoid main(void) {\n  gl_FragColor = uMaterialColor;\n}",
  lightVert:
    "attribute vec3 aPosition;\nattribute vec3 aNormal;\nattribute vec2 aTexCoord;\n\nuniform mat4 uModelViewMatrix;\nuniform mat4 uProjectionMatrix;\nuniform mat3 uNormalMatrix;\nuniform float uResolution;\nuniform int uAmbientLightCount;\nuniform int uDirectionalLightCount;\nuniform int uPointLightCount;\n\nuniform vec3 uAmbientColor[8];\nuniform vec3 uLightingDirection[8];\nuniform vec3 uDirectionalColor[8];\nuniform vec3 uPointLightLocation[8];\nuniform vec3 uPointLightColor[8];\nuniform bool uSpecular;\n\nvarying vec3 vVertexNormal;\nvarying vec2 vVertTexCoord;\nvarying vec3 vLightWeighting;\n\nvec3 ambientLightFactor = vec3(0.0, 0.0, 0.0);\nvec3 directionalLightFactor = vec3(0.0, 0.0, 0.0);\nvec3 pointLightFactor = vec3(0.0, 0.0, 0.0);\nvec3 pointLightFactor2 = vec3(0.0, 0.0, 0.0);\n\nvoid main(void){\n\n  vec4 positionVec4 = vec4(aPosition / uResolution, 1.0);\n  gl_Position = uProjectionMatrix * uModelViewMatrix * positionVec4;\n\n  vec3 vertexNormal = vec3( uNormalMatrix * aNormal );\n  vVertexNormal = vertexNormal;\n  vVertTexCoord = aTexCoord;\n\n  vec4 mvPosition = uModelViewMatrix * vec4(aPosition / uResolution, 1.0);\n  vec3 eyeDirection = normalize(-mvPosition.xyz);\n\n  float shininess = 32.0;\n  float specularFactor = 2.0;\n  float diffuseFactor = 0.3;\n\n  for(int i = 0; i < 8; i++){\n    if(uAmbientLightCount == i) break;\n    ambientLightFactor += uAmbientColor[i];\n  }\n\n  for(int j = 0; j < 8; j++){\n    if(uDirectionalLightCount == j) break;\n    vec3 dir = uLightingDirection[j];\n    float directionalLightWeighting = max(dot(vertexNormal, dir), 0.0);\n    directionalLightFactor += uDirectionalColor[j] * directionalLightWeighting;\n  }\n\n  for(int k = 0; k < 8; k++){\n    if(uPointLightCount == k) break;\n    vec3 loc = uPointLightLocation[k];\n    //loc = loc / uResolution;\n    vec3 lightDirection = normalize(loc - mvPosition.xyz);\n\n    float directionalLightWeighting = max(dot(vertexNormal, lightDirection), 0.0);\n    pointLightFactor += uPointLightColor[k] * directionalLightWeighting;\n\n    //factor2 for specular\n    vec3 reflectionDirection = reflect(-lightDirection, vertexNormal);\n    float specularLightWeighting = pow(max(dot(reflectionDirection, eyeDirection), 0.0), shininess);\n\n    pointLightFactor2 += uPointLightColor[k] * (specularFactor * specularLightWeighting\n      +  directionalLightWeighting * diffuseFactor);\n  }\n\n  if(!uSpecular){\n    vLightWeighting =  ambientLightFactor + directionalLightFactor + pointLightFactor;\n  }else{\n    vLightWeighting = ambientLightFactor + directionalLightFactor + pointLightFactor2;\n  }\n\n}\n",
  lightTextureFrag:
    "precision mediump float;\n\nuniform vec4 uMaterialColor;\nuniform sampler2D uSampler;\nuniform bool isTexture;\n\nvarying vec3 vLightWeighting;\nvarying highp vec2 vVertTexCoord;\n\nvoid main(void) {\n  if(!isTexture){\n    gl_FragColor = vec4(vec3(uMaterialColor.rgb * vLightWeighting), uMaterialColor.a);\n  }else{\n    vec4 textureColor = texture2D(uSampler, vVertTexCoord);\n    if(vLightWeighting == vec3(0., 0., 0.)){\n      gl_FragColor = textureColor;\n    }else{\n      gl_FragColor = vec4(vec3(textureColor.rgb * vLightWeighting), textureColor.a); \n    }\n  }\n}"
};
},{}]},{},[28])(28)
});