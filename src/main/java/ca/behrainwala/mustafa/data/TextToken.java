package ca.behrainwala.mustafa.data;

import java.util.*;

public class TextToken {
    private final Map<String, Integer> tokens = new HashMap<>();
    private final List<String> nonTokenText = Arrays.asList(",", ":", ".", "?", "-", "—", "!", "'", "`", "\"", ";", "’");
    private Integer nextToken=1;

    public static final String END="<$$>";

    private final Object lock = new Object();

    public void addTextToToken(String text){
        if(!nonTokenText.contains(text)){
            text = cleanText(text);
            if(!tokens.containsKey(text))
            {
                tokens.put(text, nextToken);
                synchronized (lock){
                    nextToken++;
                }
            }
        }
    }

    public int getTextToken(String text){
        text = cleanText(text);
        return tokens.get(text)==null?0:tokens.get(text);
    }

    public String getTokenText(int token){
        for(String text:tokens.keySet()){
            if(tokens.get(text)==token)
                return text;
        }

        return END;
    }

    public int getTokenSize(){
        return tokens.keySet().size();
    }

    private String cleanText(String text){
        if(text==null)
            return null;

        for(String removeChar:nonTokenText) {
            text = text.replace(removeChar, "");
        }

        return text;
    }
}
