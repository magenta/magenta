// Copyright 2017 Google Inc.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or
// implied. See the License for the specific language governing
// permissions and limitations under the License.
/**
 * Author: David Ha <hadavid@google.com>
 *
 * @fileoverview Utility functions for managing the classes of drawings
 */

function organize_class_list(good_list) {
  var good_list_mode = 0;
  if (typeof good_list === "number") {
    good_list_mode = good_list;
  }
  var list1 = [
  'cat',
  'bird',
  'bicycle',
  'octopus',
  'face',
  'flamingo',
  'cruise_ship',
  'truck',
  'pineapple',
  'spider',
  'mosquito',
  'angel',
  'butterfly',
  'pig',
  'garden',
  'the_mona_lisa',
  'crab',
  'windmill',
  'yoga'];
  var list2 = [
  'hedgehog',
  'castle',
  'catbus',
  'ant',
  'basket',
  'chair',
  'bridge',
  'diving_board',
  'firetruck',
  'flower',
  'owl',
  'palm_tree',
  'pig',
  'rain',
  'skull',
  'duck',
  'snowflake',
  'speedboat',
  'sheep',
  'scorpion',
  'sea_turtle',
  'pool',
  'paintbrush',
  'catpig',
  'bee',
  'antyoga',
  'beeflower',
  'backpack',
  'ambulance',
  'barn',
  'bus',
  'cactus',
  'calendar',
  'couch',
  'elephantpig',
  'floweryoga',
  'hand',
  'helicopter',
  'lighthouse',
  'lion',
  'parrot',
  'passport',
  'peas',
  'postcard',
  'power_outlet',
  'radio',
  'snail',
  'stove',
  'strawberry',
  'swan',
  'swing_set',
  'tiger',
  'toothpaste',
  'toothbrush',
  'trombone']
  var list3 = ['yogabicycle',
  'whale',
  'tractor',
  'radioface',
  'squirrel',
  'pigsheep',
  'lionsheep',
  'alarm_clock',
  'bear',
  'book',
  'brain',
  'bulldozer',
  'crabchair',
  'crabrabbitfacepig',
  'dog',
  'dogbunny',
  'dolphin',
  'elephant',
  'eye',
  'fan',
  'fire_hydrant',
  'frog',
  'frogsofa',
  'hedgeberry',
  'kangaroo',
  'key',
  'lantern',
  'lobster',
  'map',
  'mermaid',
  'monapassport',
  'monkey',
  'penguin',
  'rabbit',
  'rabbitturtle',
  'rhinoceros',
  'rifle',
  'roller_coaster',
  'sandwich',
  'steak'];
  list4 = ['everything']
  // list0 = shuffle(list0);
  list1 = shuffle(list1);
  list2 = shuffle(list2);
  list3 = shuffle(list3);
  if (good_list_mode === 0) {
    return list1.concat(list2.concat(list3.concat(list4)));
  } else if (good_list_mode < 4) {
    return list1;
  } else if (good_list_mode < 20) {
    return list1.concat(list2);
  }
  return list1.concat(list2.concat(list3.concat(list4)));
}

function word_in_class_list(w, class_list) {
  for (var i=0;i<class_list.length;i++) {
    if (class_list[i] == w) {
      return true;
    }
  }
  return false;
}

function shuffle(array) {
  var currentIndex = array.length, temporaryValue, randomIndex;

  // While there remain elements to shuffle...
  while (0 !== currentIndex) {

    // Pick a remaining element...
    randomIndex = Math.floor(Math.random() * currentIndex);
    currentIndex -= 1;

    // And swap it with the current element.
    temporaryValue = array[currentIndex];
    array[currentIndex] = array[randomIndex];
    array[randomIndex] = temporaryValue;
  }

  return array;
}
